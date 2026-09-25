# Resumable arbitration continuation. Trains ONLY linear IDX8 w,b; everything else frozen.
# Freezes: backbone, PLE, W_K/W_V/beta, gammas, alpha2=1.267012596130371. No R=4, no reader
# retraining, no MLP. Wall-clock 3h30 per session with atomic persistent resume checkpoints.
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import config
MARGIN_S = 600  # persist + verify margin before the hard wall
def persistent_root():
    return Path(os.environ.get(config.PERSISTENT_ENV, config.DEFAULT_PERSISTENT))
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha_file(path)
def lr_at(step, total, peak=config.ARB["lr_peak"], warm=config.ARB["warm_steps"]):
    if step < warm:
        return peak * (step + 1) / warm
    x = min((step - warm) / max(1, total - warm), 1.0)
    return peak * 0.5 * (1 + math.cos(math.pi * x))
def ordered_full(calib_dir):
    # Rebuilds the exact frozen order: base (seed 1234) + segments (seed+500/750/1000).
    from array import array as _array
    man = json.loads((Path(calib_dir) / "calibration-continuation.json").read_text())
    raws = {}
    for tag in ("500", "750", "1000"):
        fp = Path(calib_dir) / man["milestones"][tag]["file"]
        raw = fp.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == man["milestones"][tag]["sha256"]
        a = _array("I")
        a.frombytes(raw)
        raws[tag] = torch.tensor(a, dtype=torch.long).view(-1, config.SEQ)
    # Files hold cumulative blocks; longest is the full order.
    full = raws["1000"]
    assert full.shape[0] == config.MILESTONES[1000]["per_domain_blocks"] * 6
    assert full.numel() == config.MILESTONES[1000]["total"]
    return full, man
def atomic_torch_save(obj, path, probe_keys=()):
    # Milestone/resume protocol: temp file -> fsync -> reopen verification ->
    # atomic rename -> SHA. A kill at any point leaves either the old file or
    # nothing, never a half-written checkpoint.
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    probe = torch.load(str(tmp), map_location="cpu")
    for k in probe_keys:
        assert k in probe, "checkpoint missing: %s" % k
    os.replace(tmp, path)
    return sha_file(path), probe
def save_resume(root, state):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Tensors (w, b, optimizer) go in a .pt via torch.save to bytes-adjacent file.
    pt_path = root / ("resume-step%06d.pt" % state["optimizer_step"])
    pt_sha, _ = atomic_torch_save(state["pt"], pt_path, ("w", "optimizer"))
    meta = dict(state["meta"])
    meta.update({"pt_file": pt_path.name, "pt_sha256": pt_sha,
                 "total_tokens": state["total_tokens"], "optimizer_step": state["optimizer_step"]})
    mp = root / ("resume-step%06d.json" % state["optimizer_step"])
    digest = atomic_write_bytes(mp, json.dumps(meta, indent=2).encode())
    print("resume saved step=%d tokens=%d sha=%s" % (state["optimizer_step"], state["total_tokens"], pt_sha[:16]), flush=True)
    return mp, digest
def train(model, inj, hooks, arb, store, ngram_fn, full, full_man, t0, root, base_versions, ple_sig, limit_blocks=None, first_block=624):
    # Guards: only w,b trainable.
    arb.alpha2_raw.requires_grad_(False)
    # Guards: only w,b trainable.
    arb.alpha2_raw.requires_grad_(False)
    # Guards: only w,b trainable.
    arb.alpha2_raw.requires_grad_(False)
    assert arb.alpha2_raw.requires_grad is False, "alpha2 must stay frozen"
    assert abs(float(hooks.fixed_alpha2) - config.CANON["alpha2_frozen"]) < 1e-9
    tr = [p for p in arb.parameters() if p.requires_grad]
    assert len(tr) == 2 and sum(p.numel() for p in tr) == config.ARB["hidden"] + 1
    assert {id(p) for g in torch.optim.AdamW(tr, lr=1e-3).param_groups for p in g["params"]} == {id(p) for p in tr}
    opt = torch.optim.AdamW(tr, lr=config.ARB["lr_peak"], weight_decay=config.ARB["wd"])
    end_block = limit_blocks or full.shape[0]  # staged legs stop at a milestone, not 1M
    assert end_block <= full.shape[0] and end_block > first_block >= 624
    base_k = first_block - 624  # continuation never replays blocks below the init milestone
    total_steps = end_block - 624  # k runs over absolute continuation indices
    start = base_k
    # Resume from last verified persistent checkpoint if present.
    cands = sorted(Path(root).glob("resume-step*.pt")) if Path(root).exists() else []
    if cands:
        last = cands[-1]
        blob = torch.load(str(last), map_location="cpu")
        assert int(blob["cursor"]) >= first_block, "stale resume predates leg init"
        arb.w.data.copy_(blob["w"])
        arb.b.data.copy_(blob["b"])
        opt.load_state_dict(blob["optimizer"])
        start = int(blob["cursor"] - 624)
        assert blob["base_reader_sha"] == config.CANON["base_reader_sha256"]
        print("resumed from %s cursor=%d" % (last.name, blob["cursor"]), flush=True)
    model.train(False)
    leg_total = end_block - first_block  # per-leg schedule: re-warm at leg start, cosine to 0 at leg end
    for k in range(start, total_steps):
        if (time.perf_counter() - t0) > config.WALL_LIMIT_S - MARGIN_S:
            cursor = 624 + k
            state = {"optimizer_step": k, "total_tokens": cursor * config.SEQ,
                     "pt": {"w": arb.w.detach().cpu(), "b": arb.b.detach().cpu(),
                            "optimizer": opt.state_dict(), "cursor": cursor,
                            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "rng_py": random.getstate(),
                            "base_reader_sha": config.CANON["base_reader_sha256"]},
                     "meta": {"cursor_block": cursor, "calib_sha": full_man["milestones"]["1000"]["sha256"],
                              "ple_sig": ple_sig, "config": "arb-continuation-v1",
                              "alpha2_frozen": config.CANON["alpha2_frozen"],
                              "optimizer_resume_from_v49": False,
                              "init": "v49 arbitration-c weights (w,b); fresh AdamW",
                              "schedule": "per-leg re-warm 0->1e-3 over 32 steps from leg start; cosine 1e-3->0 at leg end (staged-leg precedent)",
                              "v49_note": "no v49 optimizer/scheduler state exists; continuation with newly defined schedule, not an exact optimizer resume"}}
            save_resume(root, state)
            print("WALL-GUARD: clean stop before 3h30; restart resumes", flush=True)
            return {"stopped": "wall-clock", "cursor": cursor}
        for g in opt.param_groups:
            g["lr"] = lr_at(k - base_k, leg_total)
        ids = full[624 + k].unsqueeze(0)
        mem = store.lookup(ngram_fn(ids)).to("cuda")
        hooks.set_memory(mem)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=ids.to("cuda"), use_cache=False).logits
        ce = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), ids.to("cuda")[:, 1:].reshape(-1))
        a8 = hooks.cur_alpha8
        assert a8 is not None
        reg = config.ARB["lambda_mean_reg"] * (a8.float().mean() - config.ARB["alpha8_target"]) ** 2
        loss = ce + reg
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss at continuation step %d" % k)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters())
        assert not any(p.grad is not None for p in inj.parameters())
        torch.nn.utils.clip_grad_norm_(arb.parameters(), 1.0)
        opt.step()
        if ((k - base_k) + 1) % 100 == 0:
            print("cont step %d/%d ce=%.4f mean_a8=%.4f lr=%.2e" % (k - base_k + 1, leg_total, float(ce.detach()), float(a8.detach().mean()), opt.param_groups[0]["lr"]), flush=True)
        del lg, ce, reg, loss, mem
        cur = [n for n, p in model.named_parameters() if p._version != base_versions[n]]
        assert not cur, "backbone changed"
    return {"stopped": "done", "cursor": end_block}
