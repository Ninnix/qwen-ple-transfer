# Qwengram-2B measured results

All NLLs use natural logarithms. Lower NLL is better. Accuracy is a fraction.

## Main evaluation

| Arm | Full-val NLL | Delta vs stock | Perplexity | PPL reduction | Five-domain mean | HellaSwag-1000 | LAMBADA-1000 | LAMBADA NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 2.649104 | +0.000000 | 14.141361 | 0.000% | 2.180667 | 0.456 | 0.541 | 1.825668 |
| raw | 2.607793 | -0.041311 | 13.569075 | 4.047% | 2.148208 | 0.470 | 0.532 | 1.831767 |
| linear750 | 2.611400 | -0.037704 | 13.618098 | 3.700% | 2.151835 | 0.468 | 0.546 | 1.786991 |

| Domain | Stock | Raw 15M | Linear750 | Linear750 − stock [95% CI] |
| --- | ---: | ---: | ---: | --- |
| general | 2.841681 | 2.784005 | 2.789140 | -0.052541 [-0.061059, -0.044390] |
| code | 1.361171 | 1.350415 | 1.352904 | -0.008267 [-0.012330, -0.004640] |
| math | 1.317180 | 1.298946 | 1.303235 | -0.013945 [-0.017189, -0.010753] |
| scientific | 2.073600 | 2.042347 | 2.046080 | -0.027520 [-0.032194, -0.022910] |
| multilingual | 3.309704 | 3.265325 | 3.267818 | -0.041887 [-0.047600, -0.036253] |

## Paired bootstrap

10,000 paired resamples, seed 1234: 1,024 full-val blocks, 64 blocks per domain,
and 1,000 aligned examples per benchmark. Domain-mean resampling preserves the
five-domain balance. The bootstrap code and scoring definitions are the frozen
0.8B implementation. All per-item deltas are in `arb-bootstrap.json`.

| Metric | Raw − stock | Linear750 − stock | Linear750 − raw |
| --- | --- | --- | --- |
| val_nll | -0.041311 [-0.042839, -0.039821] | -0.037704 [-0.039149, -0.036279] | +0.003606 [+0.003344, +0.003871] |
| domain_mean_nll | -0.032460 [-0.034887, -0.030002] | -0.028832 [-0.031252, -0.026388] | +0.003627 [+0.003205, +0.004045] |
| hs_acc | +0.014000 [+0.006000, +0.023000] | +0.012000 [+0.005000, +0.020000] | -0.002000 [-0.006000, +0.002000] |
| lambada_acc | -0.009000 [-0.023000, +0.005000] | +0.005000 [-0.007000, +0.017000] | +0.014000 [+0.004000, +0.024000] |
| lambada_nll | +0.006099 [-0.013367, +0.026304] | -0.038677 [-0.056742, -0.020227] | -0.044776 [-0.050949, -0.038984] |

## Arbitration and reader diagnostics

| Arm | Early alpha | Late mean | Late std | Late p05 | Late p50 | Late p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | disabled | — | — | — | — | — |
| raw | 1 | 1 | 0 | 1 | 1 | 1 |
| linear750 | 1.191673 | 0.113578 | 0.131596 | 0.008117 | 0.056448 | 0.446414 |

Late alpha is token dependent; the table uses all full-val tokens. Statistics
for every domain and benchmark are retained in `arb-summary.json`.

| Site | Trained gamma (raw and final) | Raw effective strength | Final effective strength |
| --- | ---: | ---: | ---: |
| IDX2 | 0.123204 | 0.123204 | 0.146819 |
| IDX8 | 0.178280 | 0.178280 | 0.020249 (mean) |

Stock effective strengths are zero. Reader parameter updates from the common
initialization are identical for raw and final because arbitration freezes the reader:

| Tensor | L2 update norm |
| --- | ---: |
| readers.2.beta | 0.019713 |
| readers.2.keys.0.weight | 5.618694 |
| readers.2.value.weight | 19.956890 |
| readers.8.beta | 0.042620 |
| readers.8.keys.0.weight | 3.694813 |
| readers.8.value.weight | 19.436132 |

| Site | Reader residual L2 | After arbitration L2 | Relative to hidden L2 |
| --- | ---: | ---: | ---: |
| 2 | 0.170570 | 0.203264 | 0.080500 |
| 8 | 0.296723 | 0.034950 | 0.007298 |

Residual diagnostics average the first 64 full-val blocks at the final gated
model’s hidden states. They are distinct from the parameter-update norms.

## Matched transfer controls

Each trainable arm uses 500,224 tokens, the same stream prefix, initialization,
sites, R=1 reader, AdamW and cosine schedule. DISABLED is the unchanged stock
baseline. Lower full-val NLL gives REAL > PERMUTED > RANDOM > DISABLED.

| Control | Full-val NLL | General | Code | Math | Scientific | Multilingual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DISABLED | 2.649104 | 2.841681 | 1.361171 | 1.317180 | 2.073600 | 3.309704 |
| RANDOM | 2.646467 | 2.841685 | 1.361161 | 1.316647 | 2.072199 | 3.307400 |
| PERMUTED | 2.645011 | 2.836189 | 1.360750 | 1.315739 | 2.070944 | 3.305360 |
| REAL | 2.642096 | 2.834803 | 1.359957 | 1.313944 | 2.068908 | 3.302689 |

| Full-val comparison | Paired delta [95% CI] |
| --- | --- |
| REAL_minus_PERMUTED | -0.002915 [-0.003027, -0.002803] |
| REAL_minus_RANDOM | -0.004371 [-0.004533, -0.004212] |
| REAL_minus_DISABLED | -0.007008 [-0.007308, -0.006713] |
| PERMUTED_minus_RANDOM | -0.001456 [-0.001517, -0.001398] |
| PERMUTED_minus_DISABLED | -0.004093 [-0.004292, -0.003898] |
| RANDOM_minus_DISABLED | -0.002636 [-0.002814, -0.002462] |

All six full-val intervals exclude zero in the expected direction. REAL also
beats PERMUTED and RANDOM in every domain with intervals below zero. All
domain paired comparisons are in `control-controls-500k-summary.json`.

## Execution

Kaggle provisioned two separate 16 GiB T4s. All model and reader computation
used GPU0; GPU1 had zero model allocation. Model parallelism was unnecessary
and was not benchmarked. No inter-GPU transfer overhead was incurred.

The single-T4 qualification measured 3,196.55 forward tokens/s and 810.19
training tokens/s over short benchmark loops, with 4.309/8.042 GiB forward/
training peak allocation. Actual reader-training peaks were 7.784 GiB.

| Reader segment | Tokens | Training tok/s | Training wall seconds |
| --- | ---: | ---: | ---: |
| 0→1000448 | 1000448 | 941.6 | 1062.5 |
| 1000448→5000192 | 3999744 | 942.8 | 4242.6 |
| 5000192→10000384 | 5000192 | 979.9 | 5103.0 |
| 10000384→15000064 | 4999680 | 997.3 | 5013.0 |

Training intervals come from timestamped logs, ending at reader-weight save
and excluding validation. The original 1M→5M summary divided cumulative
tokens by suffix time; the table corrects that reporting error. Raw logs remain intact.

Host RSS peaked at 13,560 MiB in reader training. Compact suffix caches used
5.701, 5.754 and 5.760 GiB; the calibration/evaluation cache used 2.199 GiB.
Linear750 trained 749,568 tokens in 686.5 s (1091.8 tokens/s), with 7.392 GiB peak GPU allocation.

Successful reader, arbitration/evaluation and control kernels took 7.071 hours total;
including the successful qualification benchmark gives 7.108 hours.
These are summed kernel wall times, including setup/cache/evaluation. They exclude
queueing, dataset uploads and interruption-related idle time. Exact environment and
per-kernel durations are in `runtime.json`.

## Persistent artifacts

Private dataset: `ninnix/qwengram-2b-checkpoints`. `milestone-shas.json` records
all checkpoint and result SHA256 values. `persistence-verification.json` records
the complete download-back verification. The 5M/10M/15M recovery checkpoints,
linear750 state, controls, per-item evaluations, bootstrap data and frozen stream
are retained. Compact manifests retain row and global-address-map hashes; caches
can be rebuilt exactly from the immutable mounted master PLE.
