# Static gates for the Lightning continuation. No GPU needed; run locally before any Studio step.
# Enforces: frozen Kaggle reader implementation, w/b-only training, frozen pins intact.
import ast
import json
from pathlib import Path
import sync_frozen
REPO = Path(__file__).resolve().parent.parent
LIG = REPO / "lightning"
def read(name):
    return (LIG / name).read_text(encoding="utf-8")
frozen = json.loads(read('frozen_shas.json'))
assert sync_frozen.blob_sha(sync_frozen.load_cells()) == frozen['frozen_blob_sha256']
print("ok [kaggle] frozen reader cells match pinned SHA")
for f in ("config.py", "sync_frozen.py", "calibration.py", "ple.py", "arb_frozen.py",
          "evaluate.py", "qualify.py", "train.py", "frozen_data.py", "frozen_build.py",
          "bootstrap.py", "studio_flow.py", "run.py"):
    src = read(f)
    ast.parse(src)
print("ok [ast] lightning modules parse")
cfg = read("config.py")
for pin in ("1.267012596130371", "078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594",
            "236dfdf285828023ca3bcd3f37366c58a3469b13", "2fc06364715b967f1860aea9cf38778875588b17",
            "26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c", "319488", "500736", "749568", "1001472"):
    assert pin in cfg, "config pin missing: %s" % pin
print("ok [config] canonical pins present")
whole = "\n".join(read(f) for f in ("train.py", "studio_flow.py", "qualify.py", "evaluate.py", "arb_frozen.py", "calibration.py", "ple.py"))
for banned in ("branches=4", "for R in [1, 4]", "model.generate", ".generate(", "MATH-500", "LiveCodeBench",
               "train_reader(", "reader-control-random", "resume.pt"):
    assert banned not in whole, "scope leak: %r" % banned
print("ok [scope] no R4 / generation / old trainer")
tr = read("train.py")
assert "alpha2_raw.requires_grad_(False)" in tr, "alpha2 must be frozen"
assert "fixed_alpha2" in tr and "1.267012596130371" in read("config.py")
assert "gamma.copy_" not in tr, "gammas must never be overwritten"
assert "inj.requires_grad_(False)" in tr or "requires_grad_(False)" in tr
assert "optimizer must contain exactly" in (REPO / "tools/arb_stage_src.py").read_text(encoding="utf-8")
print("ok [freeze] w/b-only (alpha2/gamma/reader frozen)")
assert "12600" in read("train.py") or "WALL_LIMIT_S" in tr
assert "atomic_write_bytes" in tr and "fsync" in tr and "pt_sha256" in tr
for key in ("optimizer_step", "total_tokens", "cursor", "rng_torch", "base_reader_sha"):
    assert key in tr, "resume checkpoint missing: %s" % key
print("ok [resume] 3h30 atomic persistent checkpoints with full state")
assert "h8.detach()" in read("arb_frozen.py") and "0.5 * torch.sigmoid" in read("arb_frozen.py")
assert "MemoryArbitration" in read("arb_frozen.py") and "ArbHooks" in read("arb_frozen.py")
print("ok [arb] frozen gate math reused verbatim")
cal = read("calibration.py")
assert "HELDOUT_PREFIX" in cal and "never replayed" in cal
assert "FineWeb-only" in (REPO / "tools/arb_stage_src.py").read_text(encoding="utf-8")
print("ok [calib] extension without replay, held-out preserved")
q = read("qualify.py")
assert "QUAL_TOL" in q and "qualification failed" in q
print("ok [qual] cross-backend gate aborts on drift")
assert "sigmoid*0.5" in read("studio_flow.py") and "do not launch" in read("studio_flow.py")
print("ok [decision] MLP prepared only, not launched")
assert "optimizer_resume_from_v49" in read("train.py") and "not an exact optimizer resume" in read("train.py")
print("ok [bookkeeping] v49 continuation labeled, not an optimizer resume")
assert "first_block" in read("train.py") and "never replays blocks below the init milestone" in read("train.py")
assert "first_block" in read("studio_flow.py") and "MILESTONE_BLOCKS[mtag]" in read("studio_flow.py")
print("ok [no-replay] legs train only blocks past the init milestone")
assert "atomic_torch_save" in read("train.py") and "os.replace" in read("train.py")
assert "atomic_torch_save" in read("studio_flow.py")
print("ok [atomic] milestone/resume writes are tmp-fsync-verify-rename-SHA")
print("ALL LIGHTNING CHECKS PASSED")
