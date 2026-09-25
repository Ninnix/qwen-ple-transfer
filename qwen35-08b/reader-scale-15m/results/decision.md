# REAL R=1 reader scaling: 10M to 15M

The REAL-15M R=1 IDX2+IDX8 reader with its separately calibrated linear750 IDX8 arbiter is the current canonical reader-scale candidate, identified in `../../reader-scale/results/current-canonical.json`. The prior REAL-10M canonical identity is preserved in `../../reader-scale/results/canonical-10m.json`.

The 10M reader resumed from its SHA-verified full AdamW checkpoint. The reader architecture, FineWebEdu prefix, optimizer semantics, continuation learning-rate schedule (peak 1.5e-5, 200-step warmup, cosine to the new threshold), backbone, PLE, and frozen validation artifacts matched the earlier reader-scale run. The 15M reader reached 15,000,064 tokens and was saved and tensor-reopened before its own full validation. Its raw reader full-val NLL was 2.846233, versus 2.850060 at 10M.

The 15M reader received a fresh linear dynamic IDX8 gate. Alpha2 remained fixed at 1.267012596. The balanced calibration corpus, cross-entropy plus alpha8 mean regularizer, and fresh AdamW per 500,736/749,568-token leg matched the 10M protocol. Both gate checkpoints were saved and reopened before evaluation. The 10M canonical frozen evaluation was reproduced byte-for-byte: SHA-256 `0abdd8f9eba1191beedca2a95f6483177b97b106dbc1b14a35608fdc37c250d0`.

## Frozen evaluation

| Reader + gate | Full-val NLL | Five-domain mean NLL | HellaSwag-1000 acc | LAMBADA-1000 NLL | LAMBADA-1000 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10M + linear750 | 2.857079 | 2.388370 | 0.402 | 2.236491 | 0.451 |
| 15M + new linear750 | 2.853786 | 2.384680 | 0.400 | 2.217277 | 0.456 |

The intervals below use 10,000 paired bootstrap resamples of aligned full-val blocks, benchmark examples, or 64 blocks within each domain. Domain-mean resampling preserves the five-domain balance. Negative NLL differences favor 15M. The exact per-block and per-example differences are in `contrast.json`.

| 15M minus 10M | Difference | 95% paired CI |
| --- | ---: | ---: |
| Full-val NLL | -0.003293 | [-0.003551, -0.003032] |
| Five-domain mean NLL | -0.003690 | [-0.004046, -0.003331] |
| LAMBADA NLL | -0.019215 | [-0.023367, -0.015030] |
| HellaSwag accuracy | -0.002 | [-0.005, 0.000] |
| LAMBADA accuracy | +0.005 | [-0.001, +0.012] |

| Domain NLL | 10M | 15M | Paired difference [95% CI] |
| --- | ---: | ---: | ---: |
| General | 3.138277 | 3.134094 | -0.004184 [-0.004897, -0.003386] |
| Code | 1.495856 | 1.495201 | -0.000656 [-0.001404, +0.000055] |
| Math | 1.406194 | 1.402346 | -0.003849 [-0.004912, -0.002830] |
| Scientific | 2.250134 | 2.244491 | -0.005642 [-0.006367, -0.004921] |
| Multilingual | 3.651389 | 3.647268 | -0.004120 [-0.004821, -0.003406] |

All five domain point estimates improved. General, math, scientific, and multilingual have paired CIs below zero. Code's interval crosses zero. Neither 1,000-example accuracy interval establishes a change.

## Alpha8

These means are token-weighted absolute alpha8 values. The paired mean differences and CIs resample blocks or examples, so their means can differ slightly from the token-weighted difference. `alpha8-paired.json` contains the paired per-item deltas and CIs for mean, standard deviation, p05, p50, and p95 on all eight frozen sets.

| Set | 10M mean | 15M mean | Paired item mean difference [95% CI] |
| --- | ---: | ---: | ---: |
| Full-val | 0.123088 | 0.124661 | +0.001574 [+0.001390, +0.001760] |
| HellaSwag | 0.166080 | 0.171142 | +0.005167 [+0.004947, +0.005387] |
| LAMBADA | 0.154593 | 0.162809 | +0.008225 [+0.007961, +0.008486] |
| General | 0.132050 | 0.133051 | +0.001000 [+0.000303, +0.001659] |
| Code | 0.120576 | 0.117820 | -0.002756 [-0.003451, -0.002057] |
| Math | 0.127496 | 0.128265 | +0.000768 [+0.000177, +0.001372] |
| Scientific | 0.127352 | 0.128738 | +0.001386 [+0.000923, +0.001867] |
| Multilingual | 0.124701 | 0.124943 | +0.000242 [-0.000258, +0.000720] |

| Set + reader | Std | p05 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Full-val 10M | 0.148923 | 0.006408 | 0.050890 | 0.480780 |
| Full-val 15M | 0.144834 | 0.006523 | 0.057140 | 0.471943 |
| LAMBADA 10M | 0.163154 | 0.008428 | 0.076330 | 0.490720 |
| LAMBADA 15M | 0.156654 | 0.010250 | 0.096726 | 0.482411 |

For full-val, paired item standard deviation changed by -0.004114 [-0.004210, -0.004017], p05 by +0.000095 [+0.000053, +0.000138], p50 by +0.006291 [+0.006048, +0.006534], and p95 by -0.009379 [-0.009665, -0.009090]. For LAMBADA, the corresponding differences were -0.006756 [-0.006962, -0.006559], +0.002314 [+0.002171, +0.002455], +0.020340 [+0.019699, +0.021009], and -0.010280 [-0.010709, -0.009856].

## Checkpoints and artifacts

| Artifact | SHA-256 |
| --- | --- |
| 15M pre-evaluation reader weights | `e4a760163ec07568178ab48aa533235a9af878183caf120d4e29ce1c8ce4b9dc` |
| 15M raw full optimizer checkpoint | `453fad73d7c7d5afef0d7b4f5677fc9af7ebaefa3bb329ce529ce4fd73fed426` |
| 15M bundled full optimizer resume | `7d9369cd7a1c0bdacd3604746138a09b8fd2e5d7e9ae8212507ee4976cd86229` |
| 15M linear gate at 500,736 tokens | `2dd4ffea28beba1afe2f892f5fadc686c754592409ea0e9538214a4024b74df1` |
| 15M linear gate at 749,568 tokens | `54b7a98dd5a6b0fcde69deddc8ad2efabfe86daaeca840e949c74ed3bd6d8138` |

Reader kernel: `ninnix/qwen-ple-real-r1-reader-scale-15m-p100/1`. Reader milestone dataset: `ninnix/qwen-ple-reader-15m-milestone`. Arbitration kernel: `ninnix/qwen-ple-reader-scale-15m-arbitration-p100/1`. Arbitration results dataset: `ninnix/qwen-ple-reader-scale-15m-arbitration-results`. The weights, full resume, and 750K gate were downloaded back from these datasets and their SHA-256 hashes matched the local manifests.

The paired full-val, five-domain mean, and LAMBADA NLL gains support continued NLL improvement through 15M. Code-domain NLL has a favorable point estimate but an interval crossing zero; the benchmark accuracies are unresolved at this sample size. No R4 or MLP arbiter run was part of this 15M study.
