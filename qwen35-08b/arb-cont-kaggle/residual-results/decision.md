# Residual IDX8 MLP capacity ablation

Kaggle P100 kernel `ninnix/qwen-ple-idx8-residual-mlp-ablation-p100/2` completed. The 749568-token linear gate was frozen and used as the residual baseline. Zero-initialized `Linear(32, 1)` made both alpha8 and model logits bitwise identical to linear-750K on a frozen validation block before training. Only the residual MLP's 32,833 parameters were optimized. The balanced calibration prefix was reused in its original order, with fresh AdamW and the staged 32-step warmup/cosine schedule for the 0→500736 and 500736→749568 token legs. The reader, backbone, PLE, gammas, alpha2, and linear gate remained frozen. No R4 or reader retraining ran.

The same 1024 full-val blocks, 1000 HellaSwag examples, 1000 LAMBADA examples, and five 64-block domains were scored for both MLP checkpoints. Saved P100 per-item scores from linear-750K and linear-1M on these frozen artifacts supplied the paired baselines. Both residual checkpoints were SHA-256 verified and reopened before evaluation. `checkpoints.json` records their hashes and the frozen inputs.

| Arm | Full-val NLL | Five-domain mean NLL | LAMBADA NLL | HellaSwag acc | LAMBADA acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear 750K | 2.86523215 | 2.39546272 | 2.25331764 | 0.404 | 0.451 |
| Linear 1M | 2.86520494 | 2.39543186 | 2.25338223 | 0.404 | 0.453 |
| Residual MLP 500K | 2.86521669 | 2.39544310 | 2.25358545 | 0.404 | 0.452 |
| Residual MLP 750K | 2.86518672 | 2.39538751 | 2.25357915 | 0.404 | 0.452 |

Paired differences are new minus old, with 95% bootstrap CIs (10,000 resamples):

| Contrast | Full-val NLL | Five-domain mean NLL | LAMBADA NLL |
| --- | ---: | ---: | ---: |
| MLP 500K − linear 750K | −0.00001546 [−0.00002750, −0.00000307] | −0.00001963 [−0.00004096, +0.00000145] | +0.00026780 [−0.00001185, +0.00055744] |
| MLP 500K − linear 1M | +0.00001175 [−0.00000089, +0.00002450] | +0.00001123 [−0.00000978, +0.00003303] | +0.00020321 [−0.00009584, +0.00049568] |
| MLP 750K − linear 750K | −0.00004543 [−0.00005813, −0.00003256] | −0.00007522 [−0.00009565, −0.00005526] | +0.00026150 [+0.00000095, +0.00052705] |
| MLP 750K − linear 1M | −0.00001823 [−0.00003058, −0.00000542] | −0.00004435 [−0.00006562, −0.00002344] | +0.00019691 [−0.00008062, +0.00047168] |
| MLP 750K − MLP 500K | −0.00002997 [−0.00004285, −0.00001714] | −0.00005559 [−0.00007617, −0.00003554] | −0.00000630 [−0.00030077, +0.00029267] |

Against linear-750K, residual-750K has negative paired intervals for general, math, and scientific NLL; code and multilingual intervals include zero. Against linear-1M, general and scientific intervals are negative; code, math, and multilingual include zero. The full domain details and accuracy deltas are in `contrasts.json`.

| Arm | Full-val alpha8 mean | std | p05 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear 750K | 0.121169 | 0.152115 | 0.006538 | 0.045259 | 0.486456 |
| Linear 1M | 0.120350 | 0.156790 | 0.004185 | 0.039584 | 0.489711 |
| Residual MLP 500K | 0.120716 | 0.152879 | 0.005954 | 0.043877 | 0.486673 |
| Residual MLP 750K | 0.121451 | 0.154891 | 0.005246 | 0.042859 | 0.488718 |

Residual-750K versus linear-750K changes the full-val alpha8 mean by +0.000282 [95% paired CI +0.000182, +0.000382], std by +0.002792 [+0.002719, +0.002864], p05 by −0.001405 [−0.001437, −0.001373], p50 by −0.002350 [−0.002455, −0.002246], and p95 by +0.002654 [+0.002537, +0.002777]. Per-block and per-example alpha8 changes on all eight frozen sets are in `alpha8-paired.json`.

**Readout:** the residual MLP demonstrates capacity for a small held-out full-val and five-domain mean gain at a matched 750K calibration budget, including a gain over linear-1M. It does not establish a consistent win across held-out sets: HellaSwag accuracy is unchanged and LAMBADA NLL worsens relative to linear-750K, with a nominal interval barely above zero. The 319K canonical, linear-750K, and linear-1M checkpoints remain preserved; the residual checkpoints are separate ablation artifacts. No further linear or MLP training was run.
