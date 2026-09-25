# REAL R=1 reader scaling, IDX2+IDX8

The REAL-10M R=1 reader with its separately calibrated linear750 IDX8 arbiter was the canonical candidate for this study. Its exact checkpoint identities are preserved in `canonical-10m.json`; the current canonical candidate is recorded in `current-canonical.json`.

The 5M reader was continued with its saved AdamW state to 7,500,288 and 10,000,384 tokens. The backbone, PLE, reader architecture, corpus prefix, and reader training semantics stayed fixed. The 7.5M reader came from Kaggle reader kernel v2; its full optimizer checkpoint and pre-evaluation weights were SHA and tensor verified, then mounted for the 10M continuation in v3. Both reader milestones were saved and reopened before their full-val evaluation.

| Reader | Pre-evaluation reader SHA-256 | Raw reader full-val NLL |
| --- | --- | ---: |
| 5M | `078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594` | 2.859395 |
| 7.5M | `5bd707cf0bed18e282102e5a579ad0ce53f2d7373209013b71df7fb7afe3fc29` | 2.853484 |
| 10M | `2ab87c0a17eb54b8efb0e10ac65801157ea886a8375574c670c1c30a0d213528` | 2.850060 |

Each stronger reader received a fresh linear dynamic IDX8 gate, with alpha2 fixed at 1.267012596. The balanced corpus, optimizer, schedule, alpha8 regularizer, and frozen evaluation sets matched the prior gate protocol. Checkpoints at 500,736 and 749,568 calibration tokens were SHA and reopen verified before scoring. The 5M + linear750 reference reproduced the prior P100 evaluation exactly. The 749,568 token gate gives lower paired full-val, five-domain mean, and LAMBADA NLL than the 500,736 token gate for each stronger reader, so it is used below.

## Frozen evaluation

| Reader + gate | Full-val NLL | Five-domain mean NLL | HellaSwag-1000 acc | LAMBADA-1000 NLL | LAMBADA-1000 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M + linear750 | 2.865232 | 2.395463 | 0.404 | 2.253318 | 0.451 |
| 7.5M + new linear750 | 2.860405 | 2.391695 | 0.403 | 2.244977 | 0.448 |
| 10M + new linear750 | 2.857079 | 2.388370 | 0.402 | 2.236491 | 0.451 |

| Reader + gate | General | Code | Math | Scientific | Multilingual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M + linear750 | 3.148203 | 1.498090 | 1.413835 | 2.256121 | 3.661065 |
| 7.5M + new linear750 | 3.142517 | 1.496490 | 1.411149 | 2.253543 | 3.654776 |
| 10M + new linear750 | 3.138277 | 1.495856 | 1.406194 | 2.250134 | 3.651389 |

The intervals below are 95% paired bootstrap CIs over 10,000 resamples of aligned full-val blocks, domain blocks, or benchmark examples. Negative NLL deltas favor the later reader. Five-domain mean resamples each domain's 64 blocks separately. The complete per-block and per-example deltas are in `contrasts.json`.

| Later versus earlier | Full-val NLL delta | Five-domain mean delta | LAMBADA NLL delta |
| --- | ---: | ---: | ---: |
| 7.5M versus 5M | -0.004827 [-0.005076, -0.004569] | -0.003768 [-0.004158, -0.003388] | -0.008341 [-0.012529, -0.004345] |
| 10M versus 7.5M | -0.003326 [-0.003502, -0.003155] | -0.003325 [-0.003572, -0.003082] | -0.008485 [-0.010929, -0.006086] |
| 10M versus 5M | -0.008153 [-0.008555, -0.007749] | -0.007093 [-0.007689, -0.006518] | -0.016826 [-0.023209, -0.010683] |

All five domain NLL intervals are negative for both reader steps. For 10M versus 7.5M, the paired deltas are code -0.00063 [-0.00127, -0.00003], general -0.00424 [-0.00474, -0.00372], math -0.00496 [-0.00558, -0.00434], scientific -0.00341 [-0.00399, -0.00285], and multilingual -0.00339 [-0.00380, -0.00298]. HellaSwag and LAMBADA accuracy intervals include zero; this study does not establish an accuracy gain on those 1,000-example sets.

The extra balanced calibration from 500,736 to 749,568 tokens has small but consistent paired NLL gains. At 7.5M, full-val is -0.000113 [-0.000128, -0.000097], five-domain mean is -0.000083 [-0.000110, -0.000057], and LAMBADA is -0.000839 [-0.001153, -0.000519]. At 10M, they are -0.000126 [-0.000142, -0.000109], -0.000122 [-0.000149, -0.000095], and -0.001298 [-0.001640, -0.000966].

## Alpha8

These are absolute token-level alpha8 distributions. Paired CIs for per-block or per-example alpha8 mean, standard deviation, p05, p50, and p95 on every frozen set are in `alpha8-paired.json`. The raw tensors were checked against these aggregate statistics.

| Reader + gate | Set | Mean | Std | p05 | p50 | p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5M + linear750 | Full-val | 0.121169 | 0.152115 | 0.006538 | 0.045259 | 0.486456 |
| 7.5M + new linear750 | Full-val | 0.122006 | 0.149851 | 0.006676 | 0.048657 | 0.483434 |
| 10M + new linear750 | Full-val | 0.123088 | 0.148923 | 0.006408 | 0.050890 | 0.480780 |
| 5M + linear750 | LAMBADA | 0.140633 | 0.165259 | 0.006831 | 0.054835 | 0.492176 |
| 7.5M + new linear750 | LAMBADA | 0.150319 | 0.164742 | 0.007978 | 0.068492 | 0.492544 |
| 10M + new linear750 | LAMBADA | 0.154593 | 0.163154 | 0.008428 | 0.076330 | 0.490720 |

For 10M versus 7.5M, the paired per-block full-val alpha8 mean delta is +0.001082 [0.001016, 0.001144]. The paired per-example LAMBADA mean delta is +0.004304 [0.004182, 0.004424]. The two new 750K gate weight vectors have cosine similarity 0.9931; their norms are 1.0824 and 1.0971, with biases near -1.0986. The linear gate behaves similarly on the stronger readers. No MLP gate ablation is indicated by this reader-scale comparison.

## Artifacts

- Reader milestones and SHA manifest: private Kaggle dataset `ninnix/qwen-ple-reader-scale-milestones`; reader kernel `ninnix/qwen-ple-reader-scale-7p5m-10m-p100/3`.
- New gate checkpoints, frozen evaluations, raw alpha8 tensors, paired results, and SHA manifest: private Kaggle dataset `ninnix/qwen-ple-reader-scale-arbitration-results`; arbitration kernel `ninnix/qwen-ple-reader-scale-arbitration-p100/2`.
- The canonical 319K linear arbiter, 750K balanced linear arbiter, 1M linear endpoint, and residual-MLP 750K endpoint remain preserved in `qwen35-08b/arb-cont-kaggle` and their existing Kaggle artifacts. Their local checkpoint hashes were rechecked.

Reader scaling produced consistent NLL gains through 10M under the frozen evaluation protocol. The matched linear arbiter remains sufficient for this comparison. No reader training beyond 10M, R4, or MLP arbiter run was started.
