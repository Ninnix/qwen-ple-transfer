"""Paired bootstrap 95% CIs for REAL vs {DISABLED,RANDOM,PERMUTED} on frozen full-val NLL.

Input: jsonl with one record per frozen full-val block (1024 x 512 tokens):
  {"block": 0, "n": 511, "disabled": 1523.1, "random": 1521.9, "permuted": 1520.4, "real": 1518.8}
Values are SUMMED NLL over the block's target positions (reduction='sum'),
from identical blocks under identical inference settings; pairing is by block.
Equal-size blocks => mean of per-block mean-differences reproduces dval exactly.

Usage:
  python bootstrap_nll.py perblock.jsonl
Prints mean delta, 95% percentile CI, empirical two-sided p per contrast.
Exit 0 if all three CIs exclude 0 with real < control, else exit 1.
Stdlib only. Resampling seed pinned (1234) for reproducibility.
"""
import json
import random
import statistics
import sys

CONTRASTS = (("real", "disabled"), ("real", "random"), ("real", "permuted"))
B = 10000
SEED = 1234


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert len(rows) >= 100, "need >=100 blocks, got %d" % len(rows)
    ids = [r["block"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate block ids"
    for r in rows:
        assert r["n"] > 0 and all(k in r for k in ("disabled", "random", "permuted", "real"))
    return rows


def paired_deltas(rows, a, b):
    return [(r[a] - r[b]) / r["n"] for r in rows]


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


def main(path):
    rows = load(path)
    print("blocks: %d | boot: %d | seed: %d" % (len(rows), B, SEED))
    ok = True
    for a, b in CONTRASTS:
        mean, lo, hi, p = bootstrap_ci(paired_deltas(rows, a, b))
        robust = hi < 0.0
        ok = ok and robust
        print("%s-vs-%s: mean %+.5f  95%%CI [%+.5f, %+.5f]  p=%.4g  %s"
              % (a.upper(), b.upper(), mean, lo, hi, p, "ROBUST" if robust else "NOT-ROBUST"))
    print("ALL-ROBUST" if ok else "NOT-ALL-ROBUST")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
