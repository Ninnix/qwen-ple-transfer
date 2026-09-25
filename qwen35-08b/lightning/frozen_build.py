# Frozen eval-block builders for Lightning. Verbatim logic from Kaggle c-data
# (build_validation_artifacts) + c-evalh-a2 (build_domain_blocks); only the root
# path moves to the Studio persistent dir. Every artifact is SHA-gated against
# config.REVS pins; rebuilds must reproduce byte-identical blocks or abort.
import hashlib
import json
from array import array
from pathlib import Path
import config
VAL_FAST_TOKENS = 65536
VAL_FULL_TOKENS = 524288
VAL_DATASET = "HuggingFaceFW/fineweb-edu"
VAL_CONFIG = "sample-10BT"
DOM_N_TOKENS = 32768
def _write(name, tokens_u32, meta_extra, root, kind):
    d = Path(root) / "frozen-v1"
    d.mkdir(parents=True, exist_ok=True)
    tp = d / ("tokens-%s.uint32le" % name)
    mp = d / (("validation-%s.json" % name) if kind == "val" else ("domain-%s.json" % name))
    raw = tokens_u32.tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    meta = {"name": name, "token_count": len(tokens_u32), "seq": config.SEQ,
            "count": len(tokens_u32) // config.SEQ, "tokens_sha256": digest, **meta_extra}
    if mp.exists():
        old = json.loads(mp.read_text())
        if old != meta or tp.read_bytes() != raw:
            raise RuntimeError("Immutable %s %s differs; refusing to overwrite" % (kind, name))
        print("reuse frozen %s-%s sha=%s" % (kind, name, digest[:16]), flush=True)
        return meta
    tp.write_bytes(raw)
    mp.write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote frozen %s-%s sha=%s" % (kind, name, digest[:16]), flush=True)
    return meta
def build_validation(tok, hf_token, root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    drev = api.dataset_info(VAL_DATASET).sha
    trev = api.model_info(config.REVS["target_id"]).sha
    assert trev == config.REVS["tokenizer_rev"], "tokenizer rev drift"
    need = [Path(root) / "frozen-v1" / "validation-fast.json", Path(root) / "frozen-v1" / "validation-full.json"]
    if all(p.exists() for p in need):
        return [json.loads(p.read_text()) for p in need]
    ds = load_dataset(VAL_DATASET, VAL_CONFIG, split="train", streaming=True, revision=drev)
    arr = array("I")
    docs = 0
    for row in ds:
        docs += 1
        t = row.get("text") or ""
        if t.strip():
            arr.extend(tok.encode(t, add_special_tokens=False))
            arr.append(config.ADDR["eos"])
        if len(arr) >= VAL_FULL_TOKENS:
            del arr[VAL_FULL_TOKENS:]
            break
    assert len(arr) == VAL_FULL_TOKENS, len(arr)
    base = {"dataset": VAL_DATASET, "config": VAL_CONFIG, "dataset_rev": drev,
            "target_rev": trev, "docs": docs, "skip_docs": docs}
    mf = _write("fast", array("I", arr[:VAL_FAST_TOKENS]), base, root, "val")
    mF = _write("full", arr, base, root, "val")
    assert mF["tokens_sha256"] == config.REVS["val_full_sha256"], "full-val SHA drift"
    return mf, mF
def domain_text(d, row):
    if d in ("general", "multilingual"):
        return row.get("text") or ""
    if d == "code":
        return row.get("content") or row.get("code") or row.get("text") or ""
    if d == "math":
        return (row.get("question") or "") + "\n" + (row.get("answer") or "")
    if d == "scientific":
        return (row.get("support") or "") + "\n" + (row.get("question") or "") + "\n" + (row.get("correct_answer") or row.get("answer") or "")
    return ""
def build_domain(tok, hf_token, root, domain):
    from datasets import load_dataset as _ld
    from huggingface_hub import HfApi as _Api
    import frozen_data
    mp = Path(root) / "frozen-v1" / ("domain-%s.json" % domain)
    if mp.exists():
        return frozen_data.load_blocks(Path(root) / "frozen-v1" / ("tokens-%s.uint32le" % domain),
                                       config.REVS["domains"][domain])
    errs = []
    for ds_id, cfg, split, kind in frozen_data._domain_cands(domain):
        try:
            api = _Api(token=hf_token)
            drev = api.dataset_info(ds_id).sha
            kw = {"revision": drev} if drev else {}
            ds = _ld(ds_id, split=split, streaming=True, **kw) if cfg is None else _ld(ds_id, cfg, split=split, streaming=True, **kw)
            arr = array("I")
            docs = 0
            for row in ds:
                docs += 1
                t = domain_text(domain, row)
                if t and t.strip():
                    arr.extend(tok.encode(t, add_special_tokens=False))
                    arr.append(config.ADDR["eos"])
                if len(arr) >= DOM_N_TOKENS:
                    del arr[DOM_N_TOKENS:]
                    break
            assert len(arr) == DOM_N_TOKENS, (domain, len(arr))
            _write(domain, arr, {"dataset": ds_id, "config": cfg, "split": split,
                                 "kind": kind, "dataset_rev": drev, "docs": docs}, root, "domain")
            assert json.loads(mp.read_text())["tokens_sha256"] == config.REVS["domains"][domain], "domain SHA drift: " + domain
            return frozen_data.load_blocks(Path(root) / "frozen-v1" / ("tokens-%s.uint32le" % domain),
                                           config.REVS["domains"][domain])
        except Exception as e:
            errs.append("%s: %s" % (ds_id, str(e)[:140]))
    raise RuntimeError("no domain candidate for " + domain + ": " + " | ".join(errs))
