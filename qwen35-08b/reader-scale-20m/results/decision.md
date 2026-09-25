# REAL R=1 reader scaling: 15M to 20M

The final REAL R=1 IDX2+IDX8 reader continuation reached 20,000,256 total training tokens. The reader weights were saved and tensor-reopened before full validation. The full AdamW resume was saved and reopened afterward. Backbone and PLE stayed frozen; reader architecture, FineWebEdu token prefix, optimizer, and continuation schedule matched the 15M run. The raw reader full-val NLL moved from 2.846233 at 15M to 2.843780 at 20M.

The 20M reader received a fresh linear dynamic IDX8 gate. Alpha2 stayed fixed at 1.267012596. The balanced calibration prefix, cross-entropy plus alpha8 mean regularizer, and fresh AdamW state for each 500,736/749,568-token leg matched the established protocol. Both gate checkpoints were saved, SHA-verified, and reopened before the 750K gate was evaluated. The 15M canonical evaluation reproduced byte for byte (SHA-256 `b431786fceadee6855ff196481ffe2c829e6a1c58aa9cefa09d2373ae0fc174c`).

## Frozen evaluation

| Reader + separate linear750 gate | Full-val NLL | Five-domain mean NLL | HellaSwag-1000 acc | LAMBADA-1000 NLL | LAMBADA-1000 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M | 2.865232 | 2.395463 | 0.404 | 2.253318 | 0.451 |
| 10M | 2.857079 | 2.388370 | 0.402 | 2.236491 | 0.451 |
| 15M | 2.853786 | 2.384680 | 0.400 | 2.217277 | 0.456 |
| 20M | 2.851537 | 2.383358 | 0.400 | 2.211253 | 0.458 |

Differences below are 20M minus the named reader, using 10,000 paired bootstrap resamples of aligned full-val blocks, domain blocks, or benchmark examples. The domain-mean bootstrap preserves equal weight for each of the five domains. Negative NLL differences favor 20M. [contrasts.json](contrasts.json) retains every paired block and example difference.

| Contrast | Full-val NLL difference [95% CI] | Five-domain mean difference [95% CI] | LAMBADA NLL difference [95% CI] | HellaSwag accuracy difference [95% CI] | LAMBADA accuracy difference [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20M − 15M | -0.002249 [-0.002415, -0.002085] | -0.001322 [-0.001648, -0.000996] | -0.006023 [-0.008239, -0.003736] | 0.000 [-0.003, +0.003] | +0.002 [-0.003, +0.008] |
| 20M − 10M | -0.005541 [-0.005924, -0.005151] | -0.005012 [-0.005628, -0.004391] | -0.025238 [-0.030827, -0.019676] | -0.002 [-0.006, +0.002] | +0.007 [-0.001, +0.016] |
| 20M − 5M | -0.013695 [-0.014397, -0.012973] | -0.012105 [-0.013167, -0.011047] | -0.042064 [-0.052259, -0.032239] | -0.004 [-0.010, +0.002] | +0.007 [-0.004, +0.019] |

| Domain NLL | 5M | 10M | 15M | 20M |
| --- | ---: | ---: | ---: | ---: |
| General | 3.148203 | 3.138277 | 3.134094 | 3.129606 |
| Code | 1.498090 | 1.495856 | 1.495201 | 1.495631 |
| Math | 1.413835 | 1.406194 | 1.402346 | 1.404891 |
| Scientific | 2.256121 | 2.250134 | 2.244491 | 2.241183 |
| Multilingual | 3.661065 | 3.651389 | 3.647268 | 3.645480 |

| Domain | 20M − 15M [95% CI] | 20M − 10M [95% CI] | 20M − 5M [95% CI] |
| --- | ---: | ---: | ---: |
| General | -0.004488 [-0.005163, -0.003753] | -0.008671 [-0.009855, -0.007311] | -0.018597 [-0.020726, -0.016295] |
| Code | +0.000430 [-0.000118, +0.001031] | -0.000225 [-0.001442, +0.000989] | -0.002459 [-0.005007, -0.000077] |
| Math | +0.002546 [+0.001627, +0.003420] | -0.001303 [-0.003031, +0.000358] | -0.008944 [-0.011764, -0.006260] |
| Scientific | -0.003308 [-0.003994, -0.002650] | -0.008951 [-0.010278, -0.007645] | -0.014938 [-0.017367, -0.012677] |
| Multilingual | -0.001788 [-0.002497, -0.001087] | -0.005908 [-0.007247, -0.004582] | -0.015585 [-0.017565, -0.013609] |

Full-val, five-domain mean, and LAMBADA NLL improve at 20M against all three requested baselines with paired CIs below zero. Against 15M, general, scientific, and multilingual improve; math regresses with a CI above zero, and code remains unresolved. The aggregate improvement does not erase the math regression. Neither benchmark accuracy change is established by its paired interval.

## HellaSwag trend

| Reader tokens | 5M | 7.5M | 10M | 15M | 20M |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy on the same 1,000 examples | 0.404 | 0.403 | 0.402 | 0.400 | 0.400 |

The five-point paired per-example slope is -0.000283 accuracy per million reader tokens, with 95% bootstrap CI [-0.000676, +0.000062] and two-sided bootstrap p = 0.117. From 5M to 20M, three examples became correct and seven became incorrect. The cumulative point trend is downward, but the paired interval still does not establish a regression. The 20M versus 15M point accuracy is unchanged.

## Alpha8 distribution

The values below are token-weighted absolute statistics for the freshly calibrated 20M gate. [alpha8-paired.json](alpha8-paired.json) contains paired per-block or per-example differences and CIs for all five statistics on every frozen set.

| Set | 15M mean | 20M mean | 20M std | 20M p05 | 20M p50 | 20M p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-val | 0.124661 | 0.123724 | 0.143066 | 0.006233 | 0.057927 | 0.467663 |
| HellaSwag | 0.171142 | 0.167267 | 0.165034 | 0.008802 | 0.092385 | 0.486828 |
| LAMBADA | 0.162809 | 0.157042 | 0.151759 | 0.009099 | 0.094342 | 0.472482 |
| General | 0.133051 | 0.132195 | 0.146863 | 0.008115 | 0.064916 | 0.482745 |
| Code | 0.117820 | 0.119982 | 0.133186 | 0.007731 | 0.059790 | 0.435885 |
| Math | 0.128265 | 0.128204 | 0.152828 | 0.006874 | 0.053218 | 0.475480 |
| Scientific | 0.128738 | 0.127718 | 0.152906 | 0.005421 | 0.053193 | 0.486870 |
| Multilingual | 0.124943 | 0.125074 | 0.143455 | 0.005394 | 0.059407 | 0.466429 |

| Alpha8 statistic, 20M − 15M | Full-val paired item difference [95% CI] | LAMBADA paired item difference [95% CI] |
| --- | ---: | ---: |
| Mean | -0.000938 [-0.001068, -0.000803] | -0.005907 [-0.006136, -0.005680] |
| Std | -0.001769 [-0.001852, -0.001689] | -0.005028 [-0.005206, -0.004862] |
| p05 | -0.000302 [-0.000340, -0.000264] | -0.001323 [-0.001436, -0.001211] |
| p50 | +0.000634 [+0.000434, +0.000834] | -0.002374 [-0.002960, -0.001795] |
| p95 | -0.004547 [-0.004853, -0.004255] | -0.012328 [-0.012832, -0.011838] |

The paired item means can differ from token-weighted global differences because blocks and examples contain different numbers of scored tokens.

## Checkpoints and provenance

| Artifact | SHA-256 |
| --- | --- |
| 20M pre-evaluation reader weights | `fc4bc22d7ac92bc517946f04528c8112c70835c3b000a006d17a8d3bf9cf5304` |
| 20M raw full optimizer checkpoint | `97c408ed6aaad485a80661a39488044ade59c3293b4bb79cc75f46e2cbf9531b` |
| 20M bundled full optimizer resume | `69f15d4a099db82f8f443ead9a1a54da760bc32720e419874fdaecf817111406` |
| 20M linear gate at 500,736 tokens | `b1733aec9ed9973f104b503e5ec52232d5c2b4c33d0fa15194c6389e47e30232` |
| 20M linear gate at 749,568 tokens | `e044fb8eb11430a50238757d0a39f9bd16ed414859331fec669024d7483eeb9d` |

Reader kernel: `ninnix/qwen-ple-real-r1-reader-scale-20m-p100/1`. Reader dataset: `ninnix/qwen-ple-reader-20m-milestone`. Arbitration kernel: `ninnix/qwen-ple-reader-scale-20m-arbitration-t4/1`. Arbitration dataset: `ninnix/qwen-ple-reader-scale-20m-arbitration-results`. The weights, full resume, and 750K gate were downloaded back from the private datasets and matched their SHA manifests. [reader-20m-shas.json](reader-20m-shas.json), [arb-20m-shas.json](arb-20m-shas.json), and [checkpoints.json](checkpoints.json) record the sources and frozen references.

Kaggle [retired P100 on September 15, 2026 and automatically switches P100 notebooks to T4 x2](https://www.kaggle.com/product-announcements/735239). Both the 15M and 20M kernels reported T4 compute capability 7.5 despite their P100 names. The original 20M run's 14.668 GiB lookup cache made it unusually slow: training from the continuation schedule to the pre-evaluation save took 20,551 seconds. A cache-only 20M retry used 5.708 GiB and took 3,981 seconds for the same interval. Its pre-evaluation reader weight SHA, raw full optimizer checkpoint SHA, and raw full-val NLL exactly matched the original; [reader-20m-suffix-summary.json](reader-20m-suffix-summary.json) preserves that check. This isolates the cache footprint as the practical runtime difference. The [original log](qwen-reader20-original-logs.json) and [retry log](qwen-reader20-suffix-logs.json) preserve the timing. All arbitration results above use the original reader dataset.

The current canonical reader-scale candidate remains 15M plus its separate linear750 gate, as requested before this final milestone. The 20M candidate improves aggregate held-out NLL but has a paired math-domain regression versus 15M, and HellaSwag accuracy remains unresolved. This study stops at 20M; no R4, MLP gate, reader retraining beyond 20M, or further scaling was run.
