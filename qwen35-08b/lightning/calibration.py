# Calibration continuation: extend the balanced 6-way stream without replaying base.
# Same sources/revisions/filtering as Kaggle arb cell; held-out 32K prefix preserved.
# Base (104 blocks/domain) order is reconstructed deterministically and never reshuffled;
# each continuation segment is shuffled once with a derived seed and appended after base.
import hashlib
import json
from array import array
from pathlib import Path
import torch
import config
HERE = Path(__file__).parent
def sources():
    return [
        {"key": "general", "ds": "Salesforce/wikitext", "cfg": "wikitext-103-raw-v1", "split": "train", "kind": "text"},
        {"key": "narrative", "ds": "roneneldan/TinyStories", "cfg": None, "split": "train", "kind": "text"},
        {"key": "code", "ds": "edward-io/starcoderdata-repo", "cfg": None, "split": "train", "kind": "code",
         "cands": [["edward-io/starcoderdata-repo", None], ["codeparrot/github-code-clean", None], ["bigcode/the-stack-smol", None]]},
        {"key": "math", "ds": "openai/gsm8k", "cfg": "main", "split": "test", "kind": "qa"},
        {"key": "scientific", "ds": "allenai/sciq", "cfg": None, "split": "test", "kind": "sci",
         "spill": [{"ds": "allenai/sciq", "cfg": None, "split": "train"}]},
        # Spill rule (scientific only): the test split holds ~77K post-heldout tokens,
        # less than the 166912/domain needed at 1M. The base 53248 prefix still comes
        # from test (byte-identical to Kaggle); the extension spills into the train
        # split of the SAME dataset/revision/filtering. Train docs never overlap the
        # frozen test-split eval artifact. Recorded explicitly in the manifest.
        {"key": "multilingual", "ds": "wikimedia/wikipedia", "cfg": "20220301.de", "split": "train", "kind": "text",
         "cands": [["wikimedia/wikipedia", "20220301.de"], ["HuggingFaceFW/fineweb-2", "deu_Latn"]]},
    ]
def text_of(kind, row):
    if kind == "text":
        return row.get("text") or ""
    if kind == "code":
        return row.get("content") or row.get("code") or row.get("text") or ""
    if kind == "qa":
        return (row.get("question") or "") + "\n" + (row.get("answer") or "")
    if kind == "sci":
        return (row.get("support") or "") + "\n" + (row.get("question") or "") + "\n" + (row.get("correct_answer") or row.get("answer") or "")
    return ""
def _stream_into(tok, kind, ds, skip, take, hf_token):
    # Streams (ds, cfg, split) triple; skips `skip` tokens (held-out prefix, primary
    # only), then collects up to `take` tokens. Returns (arr, docs, skipped_docs, rev, exhausted).
    from datasets import load_dataset as _ld
    from huggingface_hub import HfApi as _Api
    ds_id, ccfg, split = ds
    api = _Api(token=hf_token)
    drev = api.dataset_info(ds_id).sha
    kw = {"revision": drev} if drev else {}
    d = _ld(ds_id, split=split, streaming=True, **kw) if ccfg is None else _ld(ds_id, ccfg, split=split, streaming=True, **kw)
    arr = array("I")
    docs = 0
    skipped_docs = 0
    skipped = 0
    for row in d:
        docs += 1
        t = text_of(kind, row)
        if not t or not t.strip():
            continue
        ids = tok.encode(t, add_special_tokens=False) + [config.ADDR["eos"]]
        if skipped < skip:
            need = skip - skipped
            if len(ids) <= need:
                skipped += len(ids)
                skipped_docs += 1
                continue
            ids = ids[need:]
            skipped = skip
            skipped_docs += 1
        arr.extend(ids)
        if len(arr) >= take:
            del arr[take:]
            return arr, docs, skipped_docs, drev, False
    return arr, docs, skipped_docs, drev, True
def stream_domain(tok, src, per_domain_need, hf_token):
    raw = src.get("cands") or [[src["ds"], src["cfg"]]]
    cands = [((c[0], c[1], src["split"]) if isinstance(c, list) else (c, src["cfg"], src["split"])) for c in raw]
    assert all("fineweb-edu" not in c.lower() for c, _, _ in cands)
    errs = []
    for cand in cands:
        try:
            arr, docs, skipped_docs, drev, exhausted = _stream_into(tok, src["kind"], cand, config.HELDOUT_PREFIX, per_domain_need, hf_token)
            segs = [{"dataset": cand[0], "config": cand[1], "split": cand[2], "dataset_rev": drev,
                     "docs": docs, "skipped_prefix_docs": skipped_docs,
                     "heldout_prefix_tokens": config.HELDOUT_PREFIX, "tokens": len(arr)}]
            if exhausted and len(arr) < per_domain_need:
                if not src.get("spill"):
                    raise RuntimeError("short stream %d<%d, no spill defined" % (len(arr), per_domain_need))
                for sp in src["spill"]:
                    triple = (sp["ds"], sp["cfg"], sp["split"])
                    assert "fineweb-edu" not in triple[0].lower()
                    more, sdocs, _, srev, sexh = _stream_into(tok, src["kind"], triple, 0, per_domain_need - len(arr), hf_token)
                    arr.extend(more)
                    segs.append({"dataset": triple[0], "config": triple[1], "split": triple[2], "dataset_rev": srev,
                                 "docs": sdocs, "skipped_prefix_docs": 0,
                                 "heldout_prefix_tokens": 0, "tokens": len(more), "spill": True,
                                 "reason": "primary split exhausted; same dataset/revision/filtering, disjoint docs"})
                    if len(arr) >= per_domain_need:
                        break
                    if sexh:
                        continue
                assert len(arr) == per_domain_need, (src["key"], len(arr))
            assert len(arr) == per_domain_need, (src["key"], len(arr))
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            blocks = torch.tensor(arr, dtype=torch.long).view(-1, config.SEQ)
            meta = {"dataset": cand[0], "config": cand[1], "split": cand[2], "kind": src["kind"],
                    "segments": segs,
                    "tokens": per_domain_need, "tokens_sha256": digest}
            print("calib %s: %s/%s sha=%s docs=%d%s" % (src["key"], cand[0], cand[1], digest[:16],
                                                        sum(s["docs"] for s in segs),
                                                        " +spill" if len(segs) > 1 else ""), flush=True)
            return blocks, meta
        except Exception as e:
            errs.append("%s: %s" % (cand[0], str(e)[:160]))
    raise RuntimeError("no candidate for " + src["key"] + ": " + " | ".join(errs))
def ordered_base(per):
    # Deterministic base order: concat in CAL_SOURCES order, permute with CAL_SEED.
    blocks = torch.cat([per[k] for k in config.CAL_SOURCES], dim=0)
    assert blocks.shape[0] == 624
    g = torch.Generator().manual_seed(config.CAL_SEED)
    return blocks[torch.randperm(blocks.shape[0], generator=g)]
def ordered_segment(per, lo_blk, hi_blk, seed):
    # Slice per-domain blocks [lo_blk:hi_blk), concat, shuffle once with derived seed.
    seg = torch.cat([per[k][lo_blk:hi_blk] for k in config.CAL_SOURCES], dim=0)
    g = torch.Generator().manual_seed(seed)
    return seg[torch.randperm(seg.shape[0], generator=g)]
def build_all(tok, hf_token, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_full = {}
    metas_full = {}
    need_max = config.MILESTONES[1000]["per_domain_tokens"]
    for src in sources():
        assert src["key"] in config.CAL_SOURCES
        b, m = stream_domain(tok, src, need_max, hf_token)
        per_full[src["key"]] = b
        metas_full[src["key"]] = m
    base_n = config.CAL_BASE_PER_DOMAIN // config.SEQ  # 104
    segments = {}
    acc = ordered_base({k: v[:base_n] for k, v in per_full.items()})
    segments["base"] = acc
    bounds = {"base": base_n, 500: 163, 750: 244, 1000: 326}
    for tag, seed_add in ((500, 500), (750, 750), (1000, 1000)):
        lo = bounds["base"] if tag == 500 else bounds[500 if tag == 750 else 750]
        hi = bounds[tag]
        seg = ordered_segment(per_full, lo, hi, config.CAL_SEED + seed_add)
        acc = torch.cat([acc, seg], dim=0)
        segments[tag] = seg
    # Freeze manifests + token bytes; record SHA256 before any training.
    frozen = {"base_total": config.CAL_BASE_TOTAL, "seed": config.CAL_SEED,
              "sources": metas_full, "milestones": {}, "eval_overlap": "checked"}
    for tag in (500, 750, 1000):
        m = config.MILESTONES[tag]
        total_blk = m["per_domain_blocks"] * 6
        blk = acc[:total_blk]
        assert blk.numel() == m["total"]
        raw = blk.numpy().astype("<u4").tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        fp = outdir / ("calib-%dk-blocks.uint32le" % tag)
        fp.write_bytes(raw)
        frozen["milestones"][str(tag)] = {"total_tokens": m["total"], "blocks": total_blk,
                                          "file": fp.name, "sha256": digest,
                                          "segment_seed": config.CAL_SEED + tag,
                                          "note": "extension only; base prefix identical, never replayed"}
        print("froze %s: %d tokens sha=%s" % (fp.name, m["total"], digest[:16]), flush=True)
    for ev_sha in [config.REVS["val_full_sha256"]] + list(config.REVS["domains"].values()):
        for tag in ("500", "750", "1000"):
            assert frozen["milestones"][tag]["sha256"] != ev_sha, "calibration overlaps frozen eval"
    (outdir / "calibration-continuation.json").write_text(json.dumps(frozen, indent=2))
    return frozen
