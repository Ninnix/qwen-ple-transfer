# Qwen3.5-0.8B target-side reader adaptation (Kaggle)

Transfer the frozen n-gram PLE from `Qwen/Qwen3.8-Flash-Next-FP8`
(`e_t in R^2560`: ngram 3, 8 heads/ngram, 16 slots x 160-d, FP8 E4M3)
to the frozen text backbone of `Qwen/Qwen3.5-0.8B`
(`h_t in R^1024`, 24 layers) by training only a small target-side reader.
Backbone and PLE stay frozen.

Reader: shared-value projections (`W_K 2560->1024`, shared `W_V 2560->1024`),
contextual sigmoid gate on RMSNormed dot product, residual injection
`h = h + gamma * o`, identity init (`gamma = 0`).

## Current result and run versions (2026-09-25)

| Reader + separate linear750 gate | Role | Full-val NLL | Five-domain mean NLL | LAMBADA NLL |
| --- | --- | ---: | ---: | ---: |
| REAL-15M R=1 | Balanced canonical | 2.853786 | 2.384680 | 2.217277 |
| REAL-20M R=1 | Max-LM endpoint | 2.851537 | 2.383358 | 2.211253 |
| REAL-16M R=4, warm-started from 15M | Research endpoint; stopped early | 2.852958 | 2.384245 | 2.216757 |

The 20M R=1 reader has a paired math-domain regression against 15M. The R=4
reader improves aggregate NLL against 15M after 999,936 additional tokens,
but code and math both regress with paired intervals above zero. It did not
clearly establish balanced value from four key branches; no further R=4
training was run. The [R=1 endpoint manifest](reader-scale/results/endpoints.json)
and [R=4 report](reader-r4/results/decision.md) retain paired block/example
deltas, confidence intervals, and SHA-256 provenance.

The reader training tool now uses the [suffix-only compact PLE cache](reader-scale-20m/CACHE_IO.md)
with corpus, cache-file, and lookup-equivalence guards. The 15M→20M retry
reproduced the original reader weight SHA, raw optimizer SHA, and full-val NLL
exactly while reducing its cache from 14.668 to 5.708 GiB. The old full-prefix
builder remains as a reference. Recent reader kernels ran on Kaggle T4 x2.
The Git commit pins this tool snapshot; these Kaggle versions pin its runs:

| Reader | Reader kernel | Separate gate/evaluation kernel |
| --- | --- | --- |
| 15M R=1 | `ninnix/qwen-ple-real-r1-reader-scale-15m-p100/1` | `ninnix/qwen-ple-reader-scale-15m-arbitration-p100/1` |
| 20M R=1 | `ninnix/qwen-ple-real-r1-reader-scale-20m-p100/1` and exact suffix-cache retry `ninnix/qwen-ple-real-r1-reader-scale-20m-suffix-cache-t4/1` | `ninnix/qwen-ple-reader-scale-20m-arbitration-t4/1` |
| 16M R=4 | `ninnix/qwen-ple-real-r4-reader-warm-start-early-1m-t4/3` | `ninnix/qwen-ple-real-r4-early-arbitration-t4/1` |

The local CLI versions are pinned in [`requirements.txt`](../requirements.txt).

## Files

- `kaggle_qwen35_08b_ple_train.ipynb` — original training notebook (all run gates OFF at rest).
- `kaggle_qwen35_08b_ple_eval.ipynb` — standalone Stage A eval notebook (no training paths).
- `kaggle_qwen35_08b_ple_eval_a2.ipynb` — standalone Stage A2 eval notebook (cheap MC
  logprob + 5-domain NLL, no training paths; run by switching `code_file`).
- `kernel-metadata.json` — kernel `ninnix/qwen3-5-0-8b-ple-target-side-reader-p100`
  (historical P100 request, internet on, 11 PLE datasets attached; recent runs used T4 x2).
- `verify_notebook.py` — static/logic checks for the notebook (run locally).
- `verify_eval.py` — static checks for the Stage A eval notebook.
- `verify_eval_a2.py` — static checks for the Stage A2 eval notebook.
- `tools/` — Kaggle ops runners, log scanners, forensic scripts,
  `ple-dataset/manifest.json` (128 parts -> 33 shards).
- `alpha8/` — dynamic IDX8 gate artifacts: `alpha8-stats.json` (B/C gate
  distributions), `alpha8-diag.json` + `alpha8-diag-config.json`
  (entropy/bucket diagnostic), `alpha8-routing.json` (causal routing diagnostic).
- `arb-cont-kaggle/` — linear arbiter continuation and tiny residual MLP ablation.
- `reader-scale/`, `reader-scale-15m/`, `reader-scale-20m/` — reader scaling,
  separate linear arbitration, frozen evaluations, and paired contrasts.
- `reader-r4/` — four-branch warm start, +1M checkpoint, separate linear750
  arbitration, and early-stop report.

The dataset builder lives in `ple-data/`
(notebook + metadata). It created `ninnix/qwen38-ple-p00` … `p10`
(33 pinned FP8 shards, ~48.7 GiB, size+sha verified, one manifest per dataset).

## Addressing (do not change)

Exact source parameters, verified against the 35B pipeline and by live
tokenizer forensics: `vocab_size = 248320` (padded config size),
`eos = 248044` (`<|endoftext|>`, the PLE-training terminator —
not chat `<|im_end|>` 248046), `seed = 1234`.
Target and source BPE mappings are id-identical (source-only ids are 7
audio added-tokens above the target range: no collision).
Training streams terminate documents with raw id 248044.

## Historical run order (gated)

The original notebook gates below are historical. The current reader and
arbiter milestones are summarized above and in their study reports.

1. Smoke: 5K tokens, random memory, fast val.
2. Real-PLE smoke: 10K tokens at IDX 2, R=1 (mount required).
3. Profile per-stage wall time (lookup was 97% of step: do not sweep on `/kaggle/input`).
4. Compact cache: one sequential pass builds `compact-500k`
   (sorted addresses + FP8 rows + scale manifest, bit-exact vs MountPLE),
   reused by all placements. Do not read `/kaggle/input` randomly during training.
5. Placement sweep (SWEEP_ENABLED, DONE): IDX 2 / 8 / 2+8, R=1, 500K tokens each,
   identical tokens/seed/schedule/cache; fast val at 100K/250K, full val at 500K.
6. Calibrated controls (BRANCH_ENABLED, separate gate): winner IDX 2+8, R=1, 500K each,
   calibrated deterministic RandomPLE (per-head mean/std from real working set, no 0.06)
   + deterministic per-head bijective PermutedPLE via separate `compact-500k-perm777` cache;
   same frozen tokens/seed/schedule/val/reader-init as sweep; frozen baseline reused as DISABLED.
   No R=4, no 1M, no 5M in this run.
7. Persistence + staged REAL 1M (REAL1M_ENABLED, separate gate): checkpoints live in
   private dataset `ninnix/qwen-ple-reader-checkpoints` (canonical bundles:
   reader.safetensors/reader.json/run.json/metrics.json + resume.pt; never rely on
   `/kaggle/working`). One REAL IDX 2+8 R=1 run 0->500K (persist + reproduce
   vs -0.01362 + 4-arm downstream + per-block NLLs) then resume 500K->1M and persist.
   No R=4, no 5M.
8. Stage A2 eval (eval-only, completed): HellaSwag/PIQA/ARC-Easy/LAMBADA via
   teacher-forced logprob (sum + length-norm) + held-out NLL by domain
   (general/code/math/scientific/multilingual, 32K tokens each, SHA-frozen).
   Same five arms (REAL-500K vs 3x500K controls primary, REAL-1M exploratory),
   per-example/per-block scores persisted, paired bootstrap CIs
   (locally reproduced with `tools/bootstrap_a2.py`). No free generation.
9. Staged REAL 5M (completed as a separate gate after A2 review): continue IDX 2+8 R=1
   from 1M toward 5M, legs at 2M/3M/5M with full-val each leg; stop early if dval
   fails to improve over best by 0.001 (saturate) or worsens (reverse). Persist
   canonical bundles per leg. No R=4. Continuation LR is explicit (old 1M
   scheduler NOT restored): linear re-warm 0->1.5e-05 over 200 steps from the 1M
   point, then cosine 1.5e-05->0 at 5M. Naive restore would restart at 2.8e-05
   (94% peak shock vs 1.6e-05 for the successful 500K->1M leg); the 1.5e-05 peak
   matches that proven scale. Schedule recorded in each bundle `run.json`.

## Measured results

- Smoke: dval -7.7e-05, gamma moved, gate alive, 5.1 GiB GPU peak.
- Real-PLE smoke: PASSED (finite losses, reader grads, zero backbone grads,
  deterministic repeat lookup, +616 MiB host, 5.1 GiB GPU, checkpoint saved).
- Compact cache: 9,293,750 unique rows (dedup 1.76x), 1.419 GiB, built in ~5 min
  at ~170 MiB/s source read, row-equivalence max|diff| = 0.0.
- Lookup bench: MountPLE 37.5 tok/s vs CompactPLE 1247.9 tok/s (**33.2x**).
- 500K sweep (frozen baselines fast 2.89552 / full 2.90558, shared):

| Placement | 100K | 250K | 500K (full) |
| --- | --- | --- | --- |
| IDX 2 | -0.00147 | -0.00670 | -0.00936 |
| IDX 8 | -0.00084 | -0.00410 | -0.00587 |
| **IDX 2+8** | -0.00226 | -0.00998 | **-0.01362** |

Winner: **IDX 2+8** (best at every checkpoint, monotonic; gamma ~0.02 both sites,
gates open, largest update norms; 5.5 GiB GPU peak).
Placement signal only — random/permuted controls still required for transfer proof.
- Staged REAL 1M (v23 COMPLETE, 0 errors, IDX 2+8 R=1, same protocol/seed/schedule):
  500K leg bit-reproduced the sweep (dval -0.01362, val 2.89196, deltas <2e-06);
  1M full-val dval **-0.01910** (val 2.88648), gamma 0.025/0.028, gates stable.
  Checkpoints persisted in `ninnix/qwen-ple-reader-checkpoints` (v3).
  4-arm downstream + bootstrap CIs ran as the separate Stage A eval below.
- Stage A eval (v30 COMPLETE, 0 errors, 5 arms, frozen full-val + 3 benchmarks, ~8.2h):
  paired bootstrap 95% CIs all robust (p=0): REAL-DISABLED -0.01362
  [-0.01403,-0.01323], REAL-RANDOM -0.00940 [-0.00965,-0.00915],
  REAL-PERMUTED -0.00542 [-0.00557,-0.00527], REAL-1M-vs-REAL -0.00548
  [-0.00563,-0.00533] (locally reproduced with `tools/bootstrap_nll.py`).
  Downstream at 0.8B floor, no discrimination: MMLU-Pro-238 REAL 42/238 vs
  DISABLED 41/238 (+1 net, noise); HumanEval+ REAL 4/164 vs DISABLED 5/164
  (-1 task, noise); SimpleQA-250 0/250 all arms (strict exact-match floor,
  non-empty fluent outputs, identical length across arms). No downstream
   improvement demonstrated, no major regression either: downstream verdict is
   INCONCLUSIVE (not negative). Per protocol, MATH-500/LiveCodeBench Stage B
   does not trigger. Eval outputs persisted in dataset v5.
- Stage A2 (v34 COMPLETE, 0 errors, 5 arms, ~77 min): cheap logprob suite.
  HellaSwag-1000 (raw acc): DISABLED .393 / RANDOM .393 / PERMUTED .395 /
  REAL-500K .397 / REAL-1M .398. ARC-Easy-570: .61053 / .61053 / .61053 /
  .61930 / .62105. LAMBADA-1000 teacher-forced acc: .446 / .446 / .444 /
  .449 / .450 (NLL/tok best REAL-500K 2.25443). PIQA skipped (n=0):
  Hub scripts rejected under datasets>=4, no parquet mirror found.
  Held-out NLL (64x512 blocks, all REAL-vs-control CIs robust p=0,
  REAL-1M-vs-REAL also robust): general -0.01473/-0.01483/-0.00312/-0.00686,
  code -0.00217/-0.00219/-0.00126/-0.00090, math -0.00629/-0.00506/-0.00320/
  -0.00238, scientific -0.00559/-0.00502/-0.00235/-0.00228, multilingual
  -0.01594/-0.01101/-0.00533/-0.00657 (vs DISABLED/RANDOM/PERMUTED/REAL-1M-vs-REAL).
  MC deltas are small and mostly non-robust (HS REAL-vs-DISABLED +0.004
  p=0.03; ARC REAL-vs-PERMUTED +0.0088 p=0.017); directionally consistent but
  INCONCLUSIVE at 0.8B scale. Per-example/per-block outputs + bootstrap
  persisted; locally cross-checked (`tools/bootstrap_a2.py` compatible).
- Stage A2-5M incremental (v37 COMPLETE, 0 errors, REAL-5M only, frozen v34 reuse
  proven by exact 1e-12 reproduce of all v34 contrasts; ~22 min):
  HS-1000 REAL-5M .407 (+0.009 vs 1M p=0.003, +0.014 vs DISABLED p≈0);
  ARC-570 .6298 (+0.0088 vs 1M n.s., +0.0193 vs DISABLED p=0.019);
  LAMBADA-1000 .437 acc (-0.013 vs 1M p=0.038) with NLL/tok 2.3019
  (+0.047 vs 1M [+0.033,+0.061] — genuine reversal on this slice).
  Domain NLL all robustly better at 5M (vs 1M / vs DISABLED): general 3.1385
  (-0.0403/-0.0619), code 1.4963 (-0.0039/-0.0070), math 1.4084 (-0.0085/-0.0172),
  scientific 2.2525 (-0.0097/-0.0175), multilingual 3.6540 (-0.0361/-0.0586).
  PIQA still skipped. Merged per-example outputs persisted; no R4.
- REAL 5M (v35 COMPLETE, 0 errors, IDX 2+8 R=1, staged from dataset
  `real-1m-r1.resume.pt`, explicit continuation LR, old scheduler never restored):
  no early-stop, no NaN, no resume/backbone/PLE abort; 5.56 GiB GPU peak.
  Bundles `real-2m/3m/5m-r1` (+ working `.pt`s, `real-5m.json`) persisted to
  `ninnix/qwen-ple-reader-checkpoints` (existing bytes SHA-verified).

| Tokens | Val loss | dval vs frozen (2.90558) | Leg gain | Gamma IDX2/IDX8 |
| --- | --- | --- | --- | --- |
| 500K | 2.89196 | -0.01362 | — | ~0.02/~0.02 |
| 1M | 2.88648 | -0.01910 | -0.00548 | 0.025/0.028 |
| 2M | 2.87542 | **-0.03016** | -0.01106 | 0.037/0.042 |
| 3M | 2.86822 | **-0.03736** | -0.00720 | 0.046/0.054 |
| 5M | 2.85940 | **-0.04619** | -0.00883 | 0.062/0.075 |

  Gates open and stable (mean 0.976->0.990, std 0.134->0.064);
  grad norms 0.36/0.17/0.19; W_V update norm 13.6->15.6->18.0.
  Each leg ran re-warm (200 steps to 1.5e-05) + cosine to 0 at its own
  milestone, so recorded milestone LR is 0.0 by design. No R=4, no downstream in job.
- Stage A2-arb (v49 COMPLETE, 0 errors, frozen REAL-5M R=1 IDX 2+8, tiny ~1K-param
  arbitration on 319K balanced tokens, mean-reg toward 0.125; no R4):
  IDX2 static 1.25 vs learned 1.267 (near-identical); IDX8 dynamic gate is
  context-dependent, not a near-constant scalar (full-val mean 0.123,
  std 0.130, p05 0.019, p50 0.065, p95 0.457; wide on all sets).
  B/C beat the static-A Pareto point on val/domain retention without
  LAMBADA regression. Weights in private dataset `ninnix/qwen-ple-a2-arb`.
- Stage A2-arb-diag (v52 COMPLETE, 0 errors, inference-only, deployed-B gates
  vs disabled baseline, 147K block positions + LAMBADA spans): the gate opens
  on hard/novel/structural positions and closes on easy/local continuation —
  r(a8, entropy) +0.27 (rho +0.29), r(a8, maxprob) -0.23; easy (NLL<1) 0.097
  vs hard (NLL>4) 0.164; newline 0.272 vs punct 0.092; keyword 0.217 vs
  ident 0.083; trigram-repeat 0.102 vs novel 0.128; frequency flat.
- Stage A2-arb-route (v55 COMPLETE, 0 errors, inference-only causal routing,
  per-sequence alpha8 budget exact, learned arm reproduces arb-B to 1e-5):
  full-val NLL learned 2.86548 < hard 2.86663 < shuffled 2.86846 <
  easy 2.86942, same order on all 5 domains; LAMBADA NLL learned beats all
  reroutes (p≈0); HS flat (0.404-0.405, n.s.).

  Scientific conclusion: The dynamic memory gate learns a non-trivial
  token-level allocation policy. Memory placement itself is causal:
  preserving the same memory budget while shuffling its placement degrades
  NLL, allocating it to easy positions is consistently worst, and routing
  toward high-entropy positions recovers only part of the learned gate’s
  advantage.
