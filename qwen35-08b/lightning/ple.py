# PLE migration for Lightning. Inspect disks first; never blindly download 48.7 GiB.
# Either verify the full master PLE (same revision/shards) or build a persistent
# compact cache over exactly the required addresses. Values, addressing, hashing,
# FP8 dequant and reader semantics are unchanged (frozen_generated).
import hashlib
import json
import os
import shutil
from array import array
from pathlib import Path
import torch
import config
FULL_GIB = 48.7
FULL_MARGIN_GIB = 70.0  # full PLE + model + checkpoints + headroom
def disks(persistent, scratch="/tmp"):
    out = {}
    for name, p in (("persistent", str(persistent)), ("scratch", str(scratch))):
        try:
            u = shutil.disk_usage(p)
            out[name] = {"path": str(p), "total_gib": u.total / 1024 ** 3,
                         "free_gib": u.free / 1024 ** 3, "used_gib": u.used / 1024 ** 3}
        except Exception as e:
            out[name] = {"path": str(p), "error": str(e)[:160]}
    try:
        import torch as _t
        g = {"cuda": _t.cuda.is_available(), "torch": _t.__version__,
             "cuda_version": _t.version.cuda}
        if _t.cuda.is_available():
            g["name"] = _t.cuda.get_device_name(0)
            p = _t.cuda.get_device_properties(0)
            g["vram_total_gib"] = round(p.total_memory / 1024 ** 3, 2)
            g["capability"] = list(_t.cuda.get_device_capability(0))
        out["gpu"] = g
    except Exception as e:
        out["gpu"] = {"cuda": False, "error": str(e)[:160]}
    return out
def plan(info):
    free = info.get("persistent", {}).get("free_gib", 0) or 0
    use_full = free >= FULL_MARGIN_GIB
    mode = "full-ple" if use_full else "compact-cache"
    print("disk persistent free=%.1f GiB -> mode=%s (full needs %.0f GiB free)" % (free, mode, FULL_MARGIN_GIB), flush=True)
    return mode
def token_sequences_for_cache(calib_acc, val_full, dom_blocks, hs_ids, lam_ids):
    seqs = [calib_acc[i] for i in range(calib_acc.shape[0])]
    seqs += [val_full[i] for i in range(val_full.shape[0])]
    for d in dom_blocks:
        seqs += [dom_blocks[d][i] for i in range(dom_blocks[d].shape[0])]
    seqs += hs_ids + lam_ids
    return seqs
def uniq_addresses(seqs, ngram_fn):
    reals = []
    with torch.inference_mode():
        for s in seqs:
            reals.append(ngram_fn(s.unsqueeze(0).cpu()).reshape(-1))
    uniq = torch.unique(torch.cat(reals)).long().cpu()
    digest = hashlib.sha256(array("I", uniq.tolist()).tobytes()).hexdigest()
    print("union addresses: %d sha=%s" % (len(uniq), digest[:16]), flush=True)
    return uniq.sort().values, digest
def build_compact_from_hf(uniq, outdir, hf_token, lookup_master=None):
    # Sequential shard fetch: download one PLE shard at a time from the source
    # revision, extract needed rows, release the shard. Peak disk stays small.
    # lookup_master, if given, is a MountPLE-like object for direct extraction.
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    uh = hashlib.sha256(array("I", uniq.tolist()).tobytes()).hexdigest()
    if (outdir / "compact.json").exists():
        try:
            pm = json.loads((outdir / "compact.json").read_text())
            if pm.get("uniq_sha256") == uh and pm.get("ple_revision") == config.REVS["ple_revision"]:
                print("reuse persistent compact cache sha=%s" % uh[:16], flush=True)
                return pm
        except Exception:
            pass
    manifest = _manifest()
    assert manifest.get("ple_revision") == config.REVS["ple_revision"]
    rpp = manifest.get("rows_per_part", config.ADDR["rows_per_part"])
    parts = {int(k): v for k, v in manifest["parts"].items()}
    pa = torch.div(uniq, rpp, rounding_mode="floor")
    lo = uniq % rpp
    # Map each HF shard file to the parts it holds.
    by_file = {}
    for p, f in parts.items():
        by_file.setdefault(f, []).append(p)
    tmpl = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{p}.weight"
    scale_key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    af = (outdir / "addrs.u32").open("wb")
    rf = (outdir / "rows.u8").open("wb")
    ha = hashlib.sha256()
    hr = hashlib.sha256()
    n = 0
    prev = -1
    scale = None
    for fname in sorted(by_file):
        lp = hf_hub_download(config.REVS["source_id"], fname, revision=config.REVS["ple_revision"], token=hf_token)
        with safe_open(lp, framework="pt", device="cpu") as fh:
            if scale is None and scale_key in fh.keys():
                scale = float(fh.get_tensor(scale_key).float().mean())
            for p in sorted(by_file[fname]):
                pos = torch.nonzero(pa == p).flatten()
                if pos.numel() == 0:
                    continue
                au = uniq.index_select(0, pos)
                assert int(au[0]) > prev
                prev = int(au[-1])
                full = fh.get_slice(tmpl.format(p=p))[:]
                ab = array("I", au.tolist()).tobytes()
                rb = bytes(full.index_select(0, lo.index_select(0, pos)).view(torch.uint8).flatten().tolist())
                af.write(ab)
                rf.write(rb)
                ha.update(ab)
                hr.update(rb)
                n += pos.numel()
        try:
            os.remove(lp)
        except Exception:
            pass
    af.close()
    rf.close()
    assert scale is not None, "weight_scale not found in source shards"
    meta = {"format": "qwen-ple-compact-lightning", "version": 1, "address_count": n,
            "uniq_sha256": uh, "row_dim": 160, "scale": float(scale),
            "ple_revision": config.REVS["ple_revision"],
            "addrs_sha256": ha.hexdigest(), "rows_sha256": hr.hexdigest()}
    (outdir / "compact.json").write_text(json.dumps(meta, indent=2))
    print("compact built: %d rows scale=%g" % (n, scale), flush=True)
    return meta
def build_compact_from_local(uniq, outdir, ple_dir):
    # Same bytes as build_compact_from_hf but reads the verified full local PLE.
    from safetensors import safe_open
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    uh = hashlib.sha256(array("I", uniq.tolist()).tobytes()).hexdigest()
    if (outdir / "compact.json").exists():
        try:
            pm = json.loads((outdir / "compact.json").read_text())
            if pm.get("uniq_sha256") == uh and pm.get("ple_revision") == config.REVS["ple_revision"]:
                print("reuse persistent compact cache sha=%s" % uh[:16], flush=True)
                return pm
        except Exception:
            pass
    manifest = _manifest()
    assert manifest.get("ple_revision") == config.REVS["ple_revision"]
    rpp = manifest.get("rows_per_part", config.ADDR["rows_per_part"])
    parts = {int(k): v for k, v in manifest["parts"].items()}
    assert len(parts) == 128, "need all 128 parts, have %d" % len(parts)
    pa = torch.div(uniq, rpp, rounding_mode="floor")
    lo = uniq % rpp
    by_file = {}
    for p, f in parts.items():
        by_file.setdefault(f, []).append(p)
    tmpl = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{p}.weight"
    scale_key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    af = (outdir / "addrs.u32").open("wb")
    rf = (outdir / "rows.u8").open("wb")
    ha = hashlib.sha256()
    hr = hashlib.sha256()
    n = 0
    prev = -1
    scale = None
    for fname in sorted(by_file):
        lp = str(Path(ple_dir) / fname)
        assert Path(lp).exists(), "missing shard: %s" % fname
        with safe_open(lp, framework="pt", device="cpu") as fh:
            if scale is None and scale_key in fh.keys():
                scale = float(fh.get_tensor(scale_key).float().mean())
            for p in sorted(by_file[fname]):
                pos = torch.nonzero(pa == p).flatten()
                if pos.numel() == 0:
                    continue
                au = uniq.index_select(0, pos)
                assert int(au[0]) > prev
                prev = int(au[-1])
                full = fh.get_slice(tmpl.format(p=p))[:]
                ab = array("I", au.tolist()).tobytes()
                rb = bytes(full.index_select(0, lo.index_select(0, pos)).view(torch.uint8).flatten().tolist())
                af.write(ab)
                rf.write(rb)
                ha.update(ab)
                hr.update(rb)
                n += pos.numel()
    af.close()
    rf.close()
    assert scale is not None, "weight_scale not found in local shards"
    meta = {"format": "qwen-ple-compact-lightning", "version": 1, "address_count": n,
            "uniq_sha256": uh, "row_dim": 160, "scale": float(scale),
            "ple_revision": config.REVS["ple_revision"],
            "addrs_sha256": ha.hexdigest(), "rows_sha256": hr.hexdigest()}
    (outdir / "compact.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("compact built: %d rows scale=%g" % (n, scale), flush=True)
    return meta
def _manifest():
    here = Path(__file__).parent / "ple-manifest.json"  # transferred copy of tools/ple-dataset/manifest.json
    repo = Path(__file__).parent.parent / "tools" / "ple-dataset" / "manifest.json"
    for p in (here, repo):
        if p.exists():
            return json.loads(p.read_text())
    raise RuntimeError("ple manifest missing: transfer tools/ple-dataset/manifest.json as ple-manifest.json")
def verify_sampled(compact, master_lookup, seed=0, n=2048):
    g = torch.Generator().manual_seed(seed)
    uq = compact["uniq"] if isinstance(compact, dict) else None
    assert uq is not None
    samp = uq[torch.randint(0, len(uq), (n,), generator=g)].reshape(n // 16, 16)
    d = (compact["store"].lookup(samp) - master_lookup(samp)).abs().max().item()
    assert d == 0.0, "compact/master mismatch max|diff|=%g" % d
    print("compact equivalence ok: sampled %d rows max|diff|=0.0" % n, flush=True)
