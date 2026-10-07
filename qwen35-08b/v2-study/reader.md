# PLE reader scaling, head routing and memory utility

The canonical reader remains the balanced 15M R=1 endpoint. Its earlier
[scaling report](../reader-scale/results/decision.md) and
[endpoint manifest](../reader-scale/results/endpoints.json) preserve the
10M/15M/20M comparisons. The [20M report](../reader-scale-20m/results/decision.md)
records a paired math regression versus 15M. The
[R=4 warm-start report](../reader-r4/results/decision.md) records supported
code/math regressions after 999,936 extra reader tokens. Neither becomes
the balanced default.

## WHAT-to-read head router

A separate endpoint trains only a zero-initialized FP32 linear router,
16 x 1024 weights plus 16 biases (16,400 parameters). It maps detached
RMS-normalized IDX8 hidden states to `16 * softmax(logits)` and scales the
16 frozen reader-input head components. The weights sum to 16 per token;
uniform initialization reproduces the canonical input. Backbone, PLE,
reader, gamma/beta, alpha2 and dynamic alpha8 stay frozen. Alpha8 is bit-exact
across inference controls.

Training uses the canonical balanced calibration stream, SHA-256
`379e484148e94b08e71870e6fabe67116260d965d9420b7a7d323cb09e4bd828`,
749,568 tokens, seed 20261006, AdamW LR 1e-3, 32-step warmup, cosine to zero,
weight decay zero and gradient clip one. Only ordinary next-token CE trains
the router. The endpoint is fixed in advance; held-out evaluation does not
select checkpoints.

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| LEARNED − v1: full_val_nll | -0.002564 | [-0.002665, -0.002463] |
| LEARNED − v1: five_domain_mean_nll | -0.002716 | [-0.002888, -0.002548] |
| LEARNED − v1: lambada_nll | -0.000991 | [-0.002315, +0.000301] |
| LEARNED − v1: lambada_accuracy | +0.000000 | [-0.005000, +0.005000] |
| LEARNED − v1: hellaswag_accuracy | +0.001000 | [-0.002000, +0.005000] |

All five domains improve in the raw endpoint; full-val also beats both
HEAD-SHUFFLED and TOKEN-SHUFFLED controls. However, the fixed weight sum does
not fix reader-output amplitude. Reader RMS grows by 1.288–1.472 across the
eight evaluated datasets; full-val ratio is 1.465872, CI [1.463852,1.467832].
Every dataset breaches the predeclared material-inflation guard (lower CI
above 1.05). This prevents attributing the gain to selectivity at fixed
injection strength and blocks promotion.

Head shuffling permutes weights within each token; token shuffling permutes
routing vectors within an inference sequence. The latter may move future-derived
routing vectors to earlier positions and is a diagnostic, not a deployable policy.
The descriptive bigram routing mass is 77.8% on full validation versus 50%
uniform; its relation to alpha8 is observational, not proof of causal utility.

## Amplitude-matched causal follow-up

The exact same 749,568-token router checkpoint, SHA-256
`9af0e47d20395b7098a182362db18b4ec9ed891bb2bf3b8e1ec5bcd2d5b2c318`,
is evaluated without training. Before gamma8, per-token FP32 outputs are
normalized as `o_matched = o_routed * RMS(o_v1)/(RMS(o_routed)+1e-12)`.
Both reader-output and gamma8*alpha8 branch norms must match the canonical
branch within 1e-5; native residual rounding remains intact and is audited
separately. Alpha8 and all frozen tensor identities remain unchanged.

| Endpoint | Full-val NLL | Five-domain NLL | LAMBADA NLL | LAMBADA accuracy | HellaSwag accuracy |
| --- | --- | --- | --- | --- | --- |
| UNIFORM | 2.853786 | 2.384680 | 2.217277 | 45.6% | 40.0% |
| RAW | 2.851222 | 2.381964 | 2.216285 | 45.6% | 40.1% |
| MATCHED | 2.853946 | 2.384701 | 2.216906 | 45.5% | 39.9% |
| HEAD-SHUFFLED-MATCHED | 2.854522 | 2.385577 | 2.217537 | 45.5% | 40.0% |
| TOKEN-SHUFFLED-MATCHED | 2.854042 | 2.385083 | 2.217336 | 45.5% | 40.0% |

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| MATCHED − v1: full_val_nll | +0.000160 | [+0.000130, +0.000194] |
| MATCHED − v1: five_domain_mean_nll | +0.000021 | [-0.000031, +0.000074] |
| MATCHED − v1: general_nll | +0.000621 | [+0.000492, +0.000751] |
| MATCHED − v1: code_nll | -0.000164 | [-0.000280, -0.000059] |
| MATCHED − v1: math_nll | -0.000419 | [-0.000519, -0.000318] |
| MATCHED − v1: scientific_nll | -0.000298 | [-0.000437, -0.000159] |
| MATCHED − v1: multilingual_nll | +0.000365 | [+0.000262, +0.000472] |
| MATCHED − v1: lambada_nll | -0.000371 | [-0.000843, +0.000114] |

MATCHED still beats the matched shuffled controls, but regresses on full-val,
general and multilingual versus v1. The raw aggregate gain disappears at
canonical amplitude. This is an inference intervention on a router trained
without amplitude matching; it does not test amplitude-constrained training.
FP16 rounding is audited separately because tiny branches can show large
relative error with negligible energy. The frozen canonical endpoint is retained.

## Explicit memory-utility late gate

This historical follow-up trains a separate late linear gate (1,025 parameters)
from frozen-v1 utility labels `CE_disabled - CE_real`, aligned to the next token.
Features are RMS-normalized pre-IDX8-injection hidden states with early memory
active. Backbone, PLE, reader, gamma/beta and alpha2 stay frozen; no LoRA trains.
Canonical v1's gate is preserved as a separate artifact.

The first 512 blocks of the frozen v2 training stream provide 384 training and
128 calibration blocks, length 512, seed 20261005. Five epochs of weighted
BCEWithLogits use utility sign with weight `min(abs(utility),1)`, batch 4096
tokens, AdamW LR 1e-4 and weight decay 0.01. There is no class balancing;
the final position is excluded. Last-epoch selection is fixed, not selected
on held-out scores. These are counterfactual loss labels from the same frozen
model, not teacher-logit distillation.

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| Utility gate − v1: full_val_nll | -0.002368 | [-0.002460, -0.002279] |
| Utility gate − v1: five_domain_mean_nll | -0.001460 | [-0.001595, -0.001327] |
| Utility gate − v1: lambada_nll | +0.013150 | [+0.010745, +0.015592] |
| Utility gate − v1: lambada_accuracy | -0.002000 | [-0.008000, +0.004000] |
| Utility gate − v1: hellaswag_accuracy | +0.001000 | [+0.000000, +0.003000] |

Full-val and five-domain means improve, but LAMBADA NLL has a supported
+0.013150 regression. Gate-utility correlation does not improve. The decision
is research endpoint only, no canonical promotion. The local `v3` directory
label was an experiment name, not a released Qwengram version.

Evidence: [head analysis](results/head-router.json),
[head evaluation arrays](results/head-router/eval-LEARNED.json),
[norm-matched analysis](results/norm-matched.json),
[all norm-matched contrasts](results/norm-matched-comparisons.json),
[norm-matched arrays](results/norm-matched/eval-MATCHED.json),
[reader/branch norm audit](results/norm-matched/norms-MATCHED.json), and
[utility decision](results/utility-router.json). Frozen protocols and private
artifact receipts are preserved in `protocols/` and `persistence/`.
