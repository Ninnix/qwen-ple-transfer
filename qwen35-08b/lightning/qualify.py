# Cross-backend qualification gate. Runs before any optimizer step; aborts on drift.
# Compares fresh Lightning inference (canonical w,b + frozen alpha2) against Kaggle
# canonicals on frozen artifacts. Tolerances predeclared in config.QUAL_TOL.
import json
import time
from pathlib import Path
import config
import evaluate
def load_reference(refdir):
    refdir = Path(refdir)
    arb = json.loads((refdir / "arbitration-c.json").read_text())
    assert abs(arb["alpha2"] - config.CANON["alpha2_frozen"]) < 1e-9, "alpha2 drift"
    stats = json.loads((refdir / "alpha8-stats.json").read_text())["C"]["alpha8_by_dataset"]
    summ = json.loads((refdir / "summary.json").read_text()) if (refdir / "summary.json").exists() else None
    return arb, stats, summ
def check(name, got, ref, tol):
    d = abs(got - ref)
    print("qual %s: got=%.6f ref=%.6f |d|=%.2e tol=%.1e %s" % (name, got, ref, d, tol, "ok" if d <= tol else "FAIL"), flush=True)
    assert d <= tol, "qualification failed: %s" % name
def qualify(model, hooks, store, ngram_fn, tok, val_full, dom_blocks, hs_rows, lam_rows, refdir, subset=None):
    arb, ref_stats, ref_summ = load_reference(refdir)
    t0 = time.perf_counter()
    hard_s = 1e12  # caller bounds wall time; qualification itself is inference-only
    if subset is not None:
        val_full = val_full[:subset.get("val_blocks", val_full.shape[0])]
        hs_rows = hs_rows[:subset.get("hs", len(hs_rows))]
        lam_rows = lam_rows[:subset.get("lam", len(lam_rows))]
        dom_blocks = {d: b[:subset.get("dom_blocks", b.shape[0])] for d, b in dom_blocks.items()}
    fresh = evaluate.eval_all(model, hooks, store, ngram_fn, tok, val_full, dom_blocks, hs_rows, lam_rows, t0, hard_s)
    t = config.QUAL_TOL
    if ref_summ is not None:
        c = ref_summ.get("C-dynamic-alpha2", ref_summ.get("C", {}))
        if c.get("val_nll") is not None:
            check("val_nll", fresh["val_nll"], c["val_nll"], t["val_nll"])
        if c.get("lambada_nll_per_tok") is not None:
            check("lambada_nll", fresh["lambada_nll"], c["lambada_nll_per_tok"], t["lambada_nll"])
        for d in dom_blocks:
            if isinstance(c.get("domains"), dict) and c["domains"].get(d) is not None:
                check("domain." + d, fresh["domains"][d], c["domains"][d], t["domain_nll"])
        if t.get("acc_tol") is not None:
            for k, got, ref in (("hs_acc", fresh["hs_acc"], c.get("hs_acc")), ("lambada_acc", fresh["lambada_acc"], c.get("lambada_acc"))):
                if ref is not None:
                    check(k, got, ref, t["acc_tol"])
            print("qual acc within %.1e: HS=%.4f LAMBADA=%.4f ok" % (t["acc_tol"], fresh["hs_acc"], fresh["lambada_acc"]), flush=True)
    for ds in fresh["alpha8"]:
        if ds in ref_stats:
            for k, tol in (("mean", t["alpha8_mean"]), ("std", t["alpha8_std"]),
                           ("p05", t["alpha8_q"]), ("p50", t["alpha8_q"]), ("p95", t["alpha8_q"])):
                check("alpha8.%s.%s" % (ds, k), fresh["alpha8"][ds][k], ref_stats[ds][k], tol)
    print("QUALIFICATION PASSED", flush=True)
    return fresh
