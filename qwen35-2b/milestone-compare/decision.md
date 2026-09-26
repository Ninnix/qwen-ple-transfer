# Qwengram-2B: matched 10M versus 15M

**Canonical balanced endpoint: REAL-10M + linear750.**

Both readers are persisted milestones. Only the missing 10M arbiter was trained, using exactly the existing 749,568-token calibration protocol. The 15M arbiter and its SHA-verified evaluations were reused. No reader training was performed.

| Metric | Raw 10M | Calibrated 10M | Raw 15M | Calibrated 15M |
| --- | ---: | ---: | ---: | ---: |
| full_val_nll | 2.610445 | 2.613753 | 2.607793 | 2.611400 |
| general | 2.787902 | 2.792036 | 2.784005 | 2.789140 |
| code | 1.350796 | 1.353190 | 1.350415 | 1.352904 |
| math | 1.301128 | 1.306765 | 1.298946 | 1.303235 |
| scientific | 2.046875 | 2.050769 | 2.042347 | 2.046080 |
| multilingual | 3.268207 | 3.270495 | 3.265325 | 3.267818 |
| domain_mean_nll | 2.150981 | 2.154651 | 2.148208 | 2.151835 |
| lambada_nll | 1.836677 | 1.798536 | 1.831767 | 1.786991 |
| lambada_acc | 52.90% | 54.50% | 53.20% | 54.60% |
| hs_acc | 46.90% | 47.00% | 47.00% | 46.80% |

## Paired 15M minus 10M

Negative NLL favors 15M; positive accuracy favors 15M. Accuracy deltas are percentage points. Brackets give paired 95% confidence intervals.

| Metric | Raw delta [95% CI] | Calibrated delta [95% CI] |
| --- | ---: | ---: |
| full_val_nll | -0.002652 [-0.002875, -0.002434] | -0.002353 [-0.002562, -0.002145] |
| general | -0.003897 [-0.004779, -0.003039] | -0.002896 [-0.003573, -0.002223] |
| code | -0.000381 [-0.001154, +0.000427] | -0.000286 [-0.000964, +0.000435] |
| math | -0.002181 [-0.002886, -0.001474] | -0.003531 [-0.004136, -0.002907] |
| scientific | -0.004528 [-0.005445, -0.003654] | -0.004690 [-0.005479, -0.003919] |
| multilingual | -0.002882 [-0.003651, -0.002164] | -0.002677 [-0.003439, -0.001952] |
| domain_mean_nll | -0.002774 [-0.003133, -0.002411] | -0.002816 [-0.003129, -0.002498] |
| lambada_nll | -0.004910 [-0.008544, -0.001403] | -0.011545 [-0.014671, -0.008566] |
| lambada_acc | +0.300000 [-0.300000, +1.000000] | +0.100000 [-0.400000, +0.600000] |
| hs_acc | +0.100000 [-0.300000, +0.600000] | -0.200000 [-0.600000, +0.200000] |

## Endpoint decision

15M does not meet the predeclared balanced-improvement rule. Retain **10M + linear750** as canonical and **15M as the max-LM/research endpoint**.

15M clearly improves full-val, domain-mean and LAMBADA NLL, and four individual domains; code remains inconclusive. Neither benchmark accuracy difference is resolved. The calibrated HellaSwag decrease is **not a statistically established regression**. Choosing 10M follows the conservative selection rule, not evidence that 10M is statistically superior overall.

- hs_acc: 15M accuracy point estimate is lower

linear10: full-val perplexity 13.650184, 3.473% below stock (14.141361).

linear15: full-val perplexity 13.618098, 3.700% below stock (14.141361).

## Method and provenance

The same frozen streams and example order are used for every comparison: 1,024 full-validation blocks, 64 blocks in each of five domains, 1,000 HellaSwag examples and 1,000 LAMBADA examples. The original bootstrap uses 10,000 paired resamples with seed 1234. Domain mean is an equal-domain stratified bootstrap. Intervals are marginal, without multiplicity adjustment. Overlapping intervals do not establish equivalence, and the selected checkpoint still needs matched runtime validation.

The repeated stock evaluation passed the predeclared 2e-6 per-block/per-example NLL tolerance and identical accuracy-array checks.

See [protocol](protocol.json), [complete paired data](comparison.json), and [10M evaluation outputs](results/).
