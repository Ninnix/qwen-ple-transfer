# Linear IDX8 arbitration continuation

The 500736, 749568, and 1001472 checkpoints were SHA-256 verified and reopened before their P100 evaluations. The same frozen 1024 full-val blocks, 1000 HellaSwag examples, 1000 LAMBADA examples, and five 64-block domain sets were scored for canonical 319488 and all three milestones. The last leg used the staged fresh AdamW, 32-step warmup, cosine schedule, and frozen alpha2 = 1.267012596130371. Reader, backbone, PLE, and gammas stayed frozen.

| Tokens | Full-val NLL | Five-domain mean NLL | LAMBADA NLL | HellaSwag acc | LAMBADA acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 319488 canonical | 2.86537793 | 2.39561737 | 2.25432613 | 0.404 | 0.453 |
| 500736 | 2.86527353 | 2.39552500 | 2.25404940 | 0.404 | 0.452 |
| 749568 | 2.86523215 | 2.39546272 | 2.25331764 | 0.404 | 0.451 |
| 1001472 | 2.86520494 | 2.39543186 | 2.25338223 | 0.404 | 0.453 |

| Contrast (new − old) | Full-val NLL, paired 95% CI | Five-domain mean NLL, paired 95% CI | LAMBADA NLL, paired 95% CI |
| --- | ---: | ---: | ---: |
| 500736 − canonical 319488 | −0.000104 [−0.000118, −0.000091] | −0.000092 [−0.000116, −0.000069] | −0.000277 [−0.000600, +0.000045] |
| 749568 − 500736 | −0.000041 [−0.000055, −0.000028] | −0.000062 [−0.000086, −0.000039] | −0.000732 [−0.001057, −0.000403] |
| 1001472 − 749568 | −0.000027 [−0.000040, −0.000014] | −0.000031 [−0.000053, −0.000009] | +0.000065 [−0.000224, +0.000354] |
| 1001472 − canonical 319488 | −0.000173 [−0.000190, −0.000156] | −0.000186 [−0.000217, −0.000154] | −0.000944 [−0.001303, −0.000591] |

At 1001472, the full-val and domain-mean paired intervals still show a small gain. LAMBADA NLL is flat within its paired interval, and four of five individual domain intervals include zero. This fails the requested criterion of continued consistent improvement across held-out sets. **Declare the linear arbiter data-saturated and stop its training.** The conclusion is about the marginal 750K→1M leg; cumulative 1M scores remain better than canonical. No R=4 or reader training was run.

| Tokens | Full-val alpha8 mean | std | p05 | p50 | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 319488 canonical | 0.122468 | 0.130111 | 0.018897 | 0.065254 | 0.456518 |
| 500736 | 0.123316 | 0.142850 | 0.011838 | 0.055991 | 0.477006 |
| 749568 | 0.121169 | 0.152115 | 0.006538 | 0.045259 | 0.486456 |
| 1001472 | 0.120350 | 0.156790 | 0.004185 | 0.039584 | 0.489711 |

Paired per-block and per-example alpha8 CIs for all contrasts and sets are in `results/alpha8-paired.json`. The p95 approaches the gate's 0.5 cap; this supports inspecting capacity in a separate ablation, but the stopping decision rests on the held-out paired NLL results.

## Residual tiny-MLP ablation

`tiny_mlp.py` adds a 32,833-parameter MLP residual to the frozen 749568-token linear logit: detached h8 → RMSNorm with eps 1e-6 → Linear(1024, 32) → SiLU → Linear(32, 1), then `alpha8 = 0.5 * sigmoid(linear_logit + residual)`. The last layer starts at zero. A local tensor probe confirms bitwise equality with the linear-750K gate before training; `run_residual.py` checks both alpha8 and model logits on a real frozen validation block before its first optimizer step.

The separate P100 ablation trains only the residual MLP on the same balanced calibration prefix, saves at 500736 and 749568 calibration tokens, and compares both against the frozen linear-750K and linear-1M evaluations. It reuses the frozen reader, backbone, PLE, gammas, alpha2, evaluation inputs, AdamW settings, regularizer, and per-leg warmup/cosine schedule. Neither linear checkpoint is overwritten.
