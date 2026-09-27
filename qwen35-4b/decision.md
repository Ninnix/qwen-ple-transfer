# Qwengram-4B decision

**Balanced canonical: REAL-15M + linear750.**

Both raw endpoints and the independently calibrated 10M/15M endpoints are preserved. The backbone and PLE remained frozen; no architecture search was run.

## Decision answers

1. **REAL PLE transfer:** supported. The causal controls do resolve a REAL advantage over both PERMUTED and RANDOM.
2. **Controls:** REAL > PERMUTED > RANDOM > DISABLED (full-validation point ordering; paired intervals below).
3. **Raw perplexity reduction:** 10M 2.647%; 15M 2.970%.
4. **Calibrated perplexity reduction:** 10M 2.521%; 15M 2.784%.
5. **Canonical endpoint:** 15M under the frozen balanced rule. 15M clears both aggregate CIs without a statistically clear domain or benchmark regression.
6. **Domains:** 5/5 improve by point estimate; 5/5 have paired NLL intervals entirely below stock.
7. **Benchmarks versus stock:** HellaSwag +0.30000 [-0.50000, +1.10000] pp; LAMBADA accuracy +0.70000 [-0.40000, +1.80000] pp; LAMBADA NLL -0.01360 [-0.02674, -0.00056].
8. **Cross-scale calibrated gain:** 0.8B 5.048%; 2B 3.473%; 4B 2.784%.
9. **Execution:** single T4 qualified with decoder activation checkpointing: 9.931 GiB peak, 1388.0 forward tok/s, 406.2 training tok/s. Ordinary single-T4 backward had run out of memory.
10. **GGUF/runtime follow-up:** justified for a subsequent matched runtime validation; these results do not validate a GGUF. No export or further experiment was started.

## Frozen evaluation

| Arm | Full NLL | Δ NLL | PPL | PPL reduction | Domain mean | HS % | LAMBADA % | LAMBADA NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 2.424697 | +0.000000 | 11.298811 | 0.000% | 1.966787 | 54.80 | 65.90 | 1.350000 |
| raw10 | 2.397871 | -0.026826 | 10.999738 | 2.647% | 1.949179 | 55.20 | 66.10 | 1.343951 |
| linear10 | 2.399167 | -0.025530 | 11.014001 | 2.521% | 1.951462 | 55.10 | 66.40 | 1.337023 |
| raw15 | 2.394549 | -0.030148 | 10.963257 | 2.970% | 1.945239 | 55.30 | 66.30 | 1.346592 |
| linear15 | 2.396459 | -0.028239 | 10.984210 | 2.784% | 1.948261 | 55.10 | 66.60 | 1.336395 |

| Domain | Stock | Raw10 | Linear10 | Raw15 | Linear15 | Canonical − stock [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| general | 2.530256 | 2.500471 | 2.503567 | 2.494434 | 2.498815 | -0.03144 [-0.03636, -0.02646] |
| code | 1.221217 | 1.207320 | 1.209064 | 1.202433 | 1.204523 | -0.01669 [-0.02649, -0.00871] |
| math | 1.212598 | 1.194751 | 1.197870 | 1.191945 | 1.196319 | -0.01628 [-0.02040, -0.01231] |
| scientific | 1.894821 | 1.889946 | 1.892643 | 1.887986 | 1.891064 | -0.00376 [-0.00718, -0.00028] |
| multilingual | 2.975046 | 2.953405 | 2.954165 | 2.949395 | 2.950582 | -0.02446 [-0.02847, -0.02040] |

## Mandatory paired 15M minus 10M

Negative NLL favors 15M; positive accuracy favors 15M. Accuracy differences are percentage points.

| Metric | Raw delta [95% CI] | Calibrated delta [95% CI] |
| --- | --- | --- |
| full_val_nll | -0.00332 [-0.00354, -0.00310] | -0.00271 [-0.00292, -0.00250] |
| general | -0.00604 [-0.00702, -0.00509] | -0.00475 [-0.00560, -0.00394] |
| code | -0.00489 [-0.00683, -0.00327] | -0.00454 [-0.00613, -0.00316] |
| math | -0.00281 [-0.00355, -0.00207] | -0.00155 [-0.00233, -0.00079] |
| scientific | -0.00196 [-0.00282, -0.00110] | -0.00158 [-0.00238, -0.00078] |
| multilingual | -0.00401 [-0.00495, -0.00306] | -0.00358 [-0.00448, -0.00269] |
| domain_mean_nll | -0.00394 [-0.00445, -0.00345] | -0.00320 [-0.00366, -0.00277] |
| lambada_nll | +0.00264 [-0.00069, +0.00595] | -0.00063 [-0.00370, +0.00249] |
| lambada_acc | +0.20000 [-0.40000, +0.80000] | +0.20000 [-0.40000, +0.80000] |
| hs_acc | +0.10000 [-0.20000, +0.50000] | +0.00000 [-0.30000, +0.30000] |

The accuracy regression rule requires the paired interval to be entirely below zero. Unresolved differences do not establish equivalence. The historical 2B canonical selection used its earlier, more conservative rule and is retained independently.

## Arbitration

| Endpoint | Early α | Late mean | Std | p05 | p50 | p95 | γ early | γ late | Effective early | Effective late mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10M | 1.211371 | 0.111152 | 0.142732 | 0.008645 | 0.043879 | 0.479819 | 0.115665 | 0.151963 | 0.140113 | 0.016891 |
| 15M | 1.189445 | 0.109325 | 0.143563 | 0.007209 | 0.041232 | 0.480909 | 0.142133 | 0.202986 | 0.169060 | 0.022192 |

| Calibration − raw | Full NLL [95% CI] | Domain mean [95% CI] | HS pp [95% CI] | LAMBADA NLL [95% CI] |
| --- | --- | --- | --- | --- |
| 10M | +0.00130 [+0.00117, +0.00142] | +0.00228 [+0.00198, +0.00260] | -0.10000 [-0.50000, +0.20000] | -0.00693 [-0.00932, -0.00450] |
| 15M | +0.00191 [+0.00176, +0.00206] | +0.00302 [+0.00271, +0.00334] | -0.20000 [-0.50000, +0.00000] | -0.01020 [-0.01315, -0.00723] |

Raw reader multipliers are 1 at both sites; stock injection is disabled. Per-domain alpha distributions are preserved in the endpoint summaries.

| Reader | IDX | W_K update norm | W_V update norm | Beta update norm | Residual norm | Calibrated residual norm | Relative calibrated norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10M | 3 | 3.899831 | 19.037523 | 0.036021 | 0.177715 | 0.215279 | 0.070256 |
| 10M | 11 | 3.312719 | 16.879200 | 0.061554 | 0.216973 | 0.026926 | 0.003126 |
| 15M | 3 | 4.259596 | 20.562105 | 0.036564 | 0.232213 | 0.276205 | 0.090108 |
| 15M | 11 | 3.579591 | 19.533899 | 0.067034 | 0.334818 | 0.041144 | 0.004759 |

## Matched 500K controls

| Condition | Full NLL |
| --- | ---: |
| DISABLED | 2.424697 |
| RANDOM | 2.423902 |
| PERMUTED | 2.423692 |
| REAL | 2.422637 |

| Paired comparison | Full-val Δ NLL [95% CI] |
| --- | --- |
| REAL_minus_PERMUTED | -0.001055 [-0.001107, -0.001003] |
| REAL_minus_RANDOM | -0.001265 [-0.001326, -0.001207] |
| REAL_minus_DISABLED | -0.002060 [-0.002159, -0.001963] |
| PERMUTED_minus_RANDOM | -0.000210 [-0.000231, -0.000191] |
| PERMUTED_minus_DISABLED | -0.001006 [-0.001059, -0.000954] |
| RANDOM_minus_DISABLED | -0.000795 [-0.000850, -0.000741] |

All six paired comparisons for every held-out domain are preserved in `results/control-controls-500k-summary.json`.

## Cross-scale comparison

| Scale | Stock NLL / PPL | Canonical NLL / PPL | Δ NLL | PPL reduction | Raw reduction | Reader tokens | HS Δ pp | LAMBADA Δ NLL | Domain mean Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 2.905585 / 18.2759 | 2.853786 / 17.3534 | -0.051799 | 5.048% | 5.763% | 15000064 | +0.70 | -0.040972 | -0.037670 |
| 2B | 2.649104 / 14.1414 | 2.613753 / 13.6502 | -0.035351 | 3.473% | 3.792% | 10000384 | +1.40 | -0.027132 | -0.026016 |
| 4B | 2.424697 / 11.2988 | 2.396459 / 10.9842 | -0.028239 | 2.784% | 2.970% | 15000064 | +0.30 | -0.013604 | -0.018527 |

The measured relative benefit does decrease monotonically across these canonical checkpoints. This is a descriptive comparison of the established tracks, not an isolated causal effect of backbone size.

| Scale | Control order | Early α | Late mean / std / p05 / p50 / p95 | Train tok/s | Peak GiB |
| --- | --- | ---: | --- | ---: | ---: |
| 0.8B | REAL > PERMUTED > RANDOM > DISABLED | 1.267013 | 0.124661 / 0.144834 / 0.006523 / 0.057140 / 0.471943 | 1217.4 | 5.243 |
| 2B | REAL > PERMUTED > RANDOM > DISABLED | 1.208596 | 0.114696 / 0.141671 / 0.007410 / 0.048936 / 0.477088 | 974.5 | 7.784 |
| 4B | REAL > PERMUTED > RANDOM > DISABLED | 1.189445 | 0.109325 / 0.143563 / 0.007209 / 0.041232 / 0.480909 | 356.1 | 10.029 |

The 0.8B throughput corrects the original cumulative-token numerator to the actual continuation tokens. Throughput includes milestone validation and is not a matched hardware speed benchmark. The 2B raw 15M research endpoint separately achieves 4.047% perplexity reduction.

The cross-scale peak column is the largest per-device peak; 4B uses 1 T4 GPU(s).

## Runtime and artifacts

| Stage | Tokens/s | GPU0 GiB | GPU1 GiB | Host MiB | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| reader 0–1M | 344.0 | 10.029 | — | 10048.4 | 2907.9 |
| reader 1–5M | 373.0 | 10.029 | — | 10564.4 | 10724.5 |
| reader 5–10M | 339.7 | 10.029 | — | 10592.0 | 14721.1 |
| reader 10–15M | 356.1 | 10.029 | — | 10534.9 | 14040.8 |
| 10M arbitration | 355.8 | 9.712 | — | 10502.7 | 2107.0 |
| 15M arbitration | 385.4 | 9.712 | — | 7690.3 | 1944.9 |

Single-GPU execution has zero inter-GPU transfer overhead. Measured training throughput includes decoder recomputation during backward.

Recorded Kaggle execution wall time: at least 20.65 hours, including observed interrupted attempts, installation, cache construction, training and evaluation. Gaps between the last observed output and confirmation of session cancellation are not measured. Queue and local transfer time are excluded. Recovery restored the verified 1,000,448-token checkpoint with its optimizer, scheduler and RNG state.

Training suffix cache sizes: 5M 5.701 GiB, 10M 5.754 GiB, 15M 5.760 GiB.

The full evaluation uses 1,024 blocks, five domains of 64 blocks, and 1,000 examples per benchmark. CIs use the original 10,000 paired bootstrap resamples, seed 1234; domain mean is stratified by domain. Intervals are marginal without multiplicity correction. Repeated stock predictions passed the 2e-6 NLL tolerance and identical accuracy-array checks.

Verified checkpoints, raw per-item/per-block evaluations, bootstrap outputs and SHAs are persisted in the private Kaggle checkpoint dataset. See `results/comparison.json`, the endpoint summaries, `protocol.json`, `frozen.json`, and the runtime manifests. No further experiment follows this report.
