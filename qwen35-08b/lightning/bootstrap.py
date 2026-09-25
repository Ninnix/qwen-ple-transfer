# Paired bootstrap contrasts, verbatim logic from the Kaggle arb cell (_boot_diff).
# Resamples paired per-block/per-example deltas with replacement; two-sided p from
# the bootstrap distribution. n_boot=10000, seed=1234 unless stated.
import json
import random
N_BOOT = 10000
SEED = 1234
def boot_diff(vals_a, vals_b, n_boot=N_BOOT, seed=SEED):
    dd = [a - b for a, b in zip(vals_a, vals_b)]
    assert len(dd) > 0 and len(vals_a) == len(vals_b)
    rng = random.Random(seed)
    n = len(dd)
    reps = sorted(sum(dd[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    ge = sum(1 for x in reps if x >= 0.0)
    le = sum(1 for x in reps if x <= 0.0)
    return {"mean": sum(dd) / n, "lo": reps[int(0.025 * n_boot)], "hi": reps[int(0.975 * n_boot) - 1],
            "p": min(1.0, 2.0 * min(ge, le) / n_boot), "n": n}
def _per_tok(blocks):
    return [b["sum"] / max(1, b["n"]) for b in blocks]
def contrast(name, ea, eb):
    # ea/eb are persisted per-item eval dicts over IDENTICAL item order.
    out = {}
    out["val_nll"] = boot_diff(_per_tok(ea["val_blocks"]), _per_tok(eb["val_blocks"]))
    doms = {}
    for d in ea["dom_blocks"]:
        doms[d] = boot_diff(_per_tok(ea["dom_blocks"][d]), _per_tok(eb["dom_blocks"][d]))
    out["domains"] = doms
    out["hs_acc"] = boot_diff(ea["hs_correct"], eb["hs_correct"])
    out["lambada_acc"] = boot_diff(ea["lam_correct"], eb["lam_correct"])
    out["lambada_nll"] = boot_diff(ea["lam_nlls"], eb["lam_nlls"])
    out["name"] = name
    return out
def summarize(con):
    lines = ["contrast %s:" % con["name"]]
    for k in ("val_nll", "lambada_nll", "hs_acc", "lambada_acc"):
        m = con[k]
        lines.append("  %s: d=%+.5f [%.5f,%.5f] p=%.4g" % (k, m["mean"], m["lo"], m["hi"], m["p"]))
    for d, m in sorted(con["domains"].items()):
        lines.append("  dom %s: d=%+.5f [%.5f,%.5f] p=%.4g" % (d, m["mean"], m["lo"], m["hi"], m["p"]))
    return "\n".join(lines)
