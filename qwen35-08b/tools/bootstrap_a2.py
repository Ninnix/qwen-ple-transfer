"""Paired bootstrap 95% CIs for Stage A2: REAL-500K vs 3x500K controls + REAL-1M vs REAL.

Inputs: eval-stageA2 directory with per-example MC jsonl and per-block domain jsonl:
  hellaswag.jsonl / piqa.jsonl / arc-easy.jsonl: {"idx":..,"label":..,"real":{"correct":0/1,..},...}
  lambada.jsonl: same shape with teacher-forced correctness
  domain-<name>.jsonl: {"block":..,"n":511,"disabled":..,"random":..,"permuted":..,"real":..,"real-1m":..}
MC values are correctness (0/1); domain values are SUMMED NLL (reduction='sum').
Pairing is by example (MC) or block (domain). Seed pinned (1234).

Usage:
  python bootstrap_a2.py eval-stageA2
Prints mean delta, 95% percentile CI, empirical two-sided p per contrast.
Exit 0 on valid inputs (report-only; A2 can be inconclusive on a small base).
Stdlib only.
"""
import json
import random
import statistics
import sys
from pathlib import Path

CONTRASTS = (("real", "disabled"), ("real", "random"), ("real", "permuted"), ("real-1m", "real"))
B = 10000
SEED = 1234
MC_TASKS = ("hellaswag", "piqa", "arc-easy", "lambada")
DOMAINS = ("general", "code", "math", "scientific", "multilingual")


def bootstrap_ci(deltas, n_boot=B, seed=SEED):
    rng = random.Random(seed)
    n = len(deltas)
    mean = statistics.fmean(deltas)
    reps = []
    for _ in range(n_boot):
        reps.append(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)))
    reps.sort()
    lo = reps[int(0.025 * n_boot)]
    hi = reps[int(0.975 * n_boot) - 1]
    ge = sum(1 for x in reps if x >= 0.0)
    le = sum(1 for x in reps if x <= 0.0)
    p = 2.0 * min(ge, le) / n_boot
    return mean, lo, hi, min(p, 1.0)


def load_mc(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, "empty %s" % path
    for r in rows:
        assert "label" in r
        for a, b in CONTRASTS:
            assert a in r and b in r and "correct" in r[a] and "correct" in r[b], (path, r.get("idx"))
    return rows


def load_domain(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 16, "need >=16 blocks, got %d in %s" % (len(rows), path)
    for r in rows:
        assert r["n"] > 0 and all(k in r for k in ("disabled", "random", "permuted", "real", "real-1m"))
    return rows


def main(d):
    d = Path(d)
    for t in MC_TASKS:
        rows = load_mc(d / f"{t}.jsonl")
        print("%s: n=%d" % (t, len(rows)))
        for a, b in CONTRASTS:
            va = [r[a]["correct"] for r in rows]
            vb = [r[b]["correct"] for r in rows]
            mean, lo, hi, p = bootstrap_ci([x - y for x, y in zip(va, vb)])
            print("%s %s-vs-%s: dacc %+.4f 95%%CI [%+.4f, %+.4f] p=%.4g %s"
                  % (t, a.upper(), b.upper(), mean, lo, hi, p, "ROBUST" if lo > 0 or hi < 0 else "INCONCLUSIVE"))
    for dom in DOMAINS:
        p = d / f"domain-{dom}.jsonl"
        if not p.exists():
            print("domain-%s: missing, skip" % dom)
            continue
        rows = load_domain(p)
        print("domain-%s: blocks=%d" % (dom, len(rows)))
        for a, b in CONTRASTS:
            dd = [(r[a] - r[b]) / r["n"] for r in rows]
            mean, lo, hi, pval = bootstrap_ci(dd)
            print("domain-%s %s-vs-%s: mean %+.5f 95%%CI [%+.5f, %+.5f] p=%.4g %s"
                  % (dom, a.upper(), b.upper(), mean, lo, hi, pval, "ROBUST" if hi < 0 else "NOT-ROBUST"))
    print("A2-BOOTSTRAP-DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
