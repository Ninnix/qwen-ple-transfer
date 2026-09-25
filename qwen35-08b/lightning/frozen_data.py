# Frozen data builders for Lightning. Same datasets, configs, splits, candidate order,
# text extraction and SHA gates as Kaggle c-evalh-a2 + arb calibration. Only the root
# path moves from /kaggle/working to the Studio persistent dir (infrastructure-only).
import hashlib
import json
from array import array
from pathlib import Path
import config
DOMAINS = ("general", "code", "math", "scientific", "multilingual")
EVAL_SEED = 1234
def _ds_rev(ds_id, hf_token):
    from huggingface_hub import HfApi
    try:
        return HfApi(token=hf_token).dataset_info(ds_id).sha
    except Exception:
        return None
def _load_split(ds_id, split, hf_token, cfg=None):
    from datasets import load_dataset
    rev = _ds_rev(ds_id, hf_token)
    kw = {"revision": rev} if rev else {}
    if cfg is None:
        return load_dataset(ds_id, split=split, **kw), rev, None
    return load_dataset(ds_id, cfg, split=split, **kw), rev, cfg
def _subset(rows, n, seed=EVAL_SEED):
    import random
    rows = list(rows)
    if len(rows) <= n:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]
def _load_first(cands, split, hf_token):
    errs = []
    for cid, ccfg in cands:
        try:
            ds, rev, _ = _load_split(cid, split, hf_token, ccfg)
            print("dataset resolved: " + cid + ("/" + ccfg if ccfg else ""), flush=True)
            return ds, rev, cid, ccfg
        except Exception as e:
            errs.append(cid + ": " + str(e)[:120])
    raise RuntimeError("no candidate resolved: " + " | ".join(errs))
def hellaswag_rows(n, hf_token):
    ds, rev, ds_id, cfg = _load_first([("Rowan/hellaswag", None)], "validation", hf_token)
    rows = []
    for r in ds:
        ctx = r.get("ctx") or ((r.get("ctx_a") or "") + " " + (r.get("ctx_b") or "")).strip()
        ends = list(r.get("endings") or [])
        try:
            lab = int(str(r.get("label")).strip())
        except Exception:
            continue
        if not ctx or len(ends) != 4:
            continue
        rows.append({"ctx": ctx, "endings": ends, "label": lab})
    rows = _subset(rows, n)
    assert config.REVS["hellaswag"]["rev"] == str(rev), "hellaswag rev drift"
    return rows, {"dataset": ds_id, "config": cfg, "rev": str(rev), "n": len(rows)}
def lambada_rows(n, hf_token):
    ds, rev, ds_id, cfg = _load_first([("EleutherAI/lambada_openai", None)], "test", hf_token)
    rows = []
    for r in ds:
        t = (r.get("text") or "").strip()
        parts = t.split()
        if len(parts) < 10:
            continue
        rows.append({"context": " ".join(parts[:-1]), "target": parts[-1], "text": t})
    rows = _subset(rows, n)
    assert config.REVS["lambada"]["rev"] == str(rev), "lambada rev drift"
    return rows, {"dataset": ds_id, "config": cfg, "rev": str(rev), "n": len(rows)}
def _domain_cands(d):
    # Fully-qualified repo IDs only (newer datasets releases reject bare aliases
    # like wikitext/wikipedia; aliases resolve to these same repos, so pinned SHAs still gate).
    if d == "general":
        return [("Salesforce/wikitext", "wikitext-103-raw-v1", "test", "text"), ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", "text")]
    if d == "code":
        return [("edward-io/starcoderdata-repo", None, "train", "code"), ("codeparrot/github-code-clean", None, "train", "code"), ("bigcode/the-stack-smol", None, "train", "code")]
    if d == "math":
        return [("openai/gsm8k", "main", "test", "qa")]
    if d == "scientific":
        return [("allenai/sciq", None, "test", "sci")]
    if d == "multilingual":
        return [("wikimedia/wikipedia", "20220301.de", "train", "text"), ("HuggingFaceFW/fineweb-2", "deu_Latn", "train", "text")]
    raise ValueError(d)
def _domain_text(d, row):
    if d in ("general", "multilingual"):
        return row.get("text") or ""
    if d == "code":
        return row.get("content") or row.get("code") or row.get("text") or ""
    if d == "math":
        return (row.get("question") or "") + "\n" + (row.get("answer") or "")
    if d == "scientific":
        return (row.get("support") or "") + "\n" + (row.get("question") or "") + "\n" + (row.get("correct_answer") or row.get("answer") or "")
    return ""
def frozen_dir(root):
    d = Path(root) / "frozen-v1"
    d.mkdir(parents=True, exist_ok=True)
    return d
def load_blocks(path, expect_sha):
    import torch
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expect_sha, "SHA drift: %s" % path
    a = array("I")
    a.frombytes(raw)
    return torch.tensor(a, dtype=torch.long).view(-1, config.SEQ)
