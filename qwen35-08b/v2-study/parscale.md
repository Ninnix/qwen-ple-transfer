# Official-style ParScale P=2: final matched 500K ablation

**A is a useful ParScale-only research endpoint. B is a negative-combination
result. Qwengram v1 remains canonical.** Both arms stop at 500,224 training
tokens; the final full frozen evaluation uses those already persisted
checkpoints and performs no training. No 1M/5M continuation or P=4 run follows.

## Architecture and frozen scope

The architectural reference is [QwenLM/ParScale](https://github.com/QwenLM/ParScale),
commit `cd6acb48ba6d3b1f9715785ff0d212751210177d`, and its
[paper](https://arxiv.org/abs/2505.10475). The port duplicates the token input
into two streams through the same frozen Qwen3.5 backbone, with 48 learned
stream-specific K/V prefix slots at attention layers and learned dynamic
weighted aggregation of final hidden states. The aggregation is
Linear(2048,1024), SiLU, Linear(1024,2), FP32 softmax with 0.01 smoothing,
official hidden-by-stream feature order, after final norm and before frozen
lm_head. This is a Qwen3.5 adaptation, not a native Qwen2 implementation.

Full attention adds learned prefix K/V in native cache coordinates after
query/key norms and partial RoPE, keeping all prefix slots visible and the
native output gate/projection. Gated DeltaNet initializes its native normalized
recurrence with learned virtual K/V, beta one and zero decay, before real
tokens. It adds no virtual query or convolution input and retains native
convolution and gates. These hybrid-attention choices are part of the port
being tested; a 500K result does not establish outcomes for other ports/budgets.

B uses identical token IDs, addresses and frozen raw PLE rows in both streams.
Only raw lookup is shared. Reader queries, gating, canonical IDX2/IDX8 injection
and alpha8 are evaluated independently from each stream's own hidden state.
The final reader output is never shared between streams.

Only 52 ParScale tensors, 9,767,938 parameters, enter AdamW: stream K/V prefixes
and aggregation weights/biases. The [exact optimizer list](protocols/parscale-optimizer-parameters.json)
preserves names, shapes and counts. Backbone, embeddings, lm_head, PLE, reader,
gammas/beta, alpha2 and existing alpha8 arbiter remain frozen. A and B use
identical initialization, ParScale shapes, training stream, successful-token
budget and schedule. AdamW uses peak LR 3e-4, betas (0.9,0.95), epsilon 1e-8,
weight decay 0.1, 100-step warmup, cosine toward 1e-5 on a fixed 5,000,192-token
horizon, gradient clip one, FP16 backbone and FP32 trainable modules/CE with
GradScaler. The horizon is a schedule parameter, not completed training.
The input stream is the exact modern balanced DeltaPLE stream prefix, with
the same exclusions/pins and ordinary next-token CE.

## Identity and qualification

Zero prefix values, suppressed full-attention prefix keys fitted from queries
on the first 20 training blocks of both frozen baselines, and zero final
aggregation projection give a controlled near-base initialization. This
query-only setup does not train backbone/reader or use teacher-logit targets.

| Init versus its base | Mean probe NLL delta | Max absolute block NLL delta | LAMBADA probe NLL delta |
| --- | ---: | ---: | ---: |
| A versus stock | +0.000578 | 0.003196 | -0.001432 |
| B versus canonical v1 | +0.000336 | 0.003947 | +0.001511 |

Both independent checks pass the predeclared mean 0.005/block 0.02 tolerance.
These are initialization probes, not the final full-suite deltas.

| Qualification | Input tok/s | Peak allocated VRAM GiB | Peak host RAM GiB |
| --- | ---: | ---: | ---: |
| A, no checkpointing | 794.23 | 7.343 | 2.971 |
| B, no checkpointing | 736.90 | 7.385 | 9.958 |
| A, checkpointing | 597.50 | 3.326 | 9.972 |
| B, checkpointing | 561.44 | 3.339 | 9.972 |

Each optimizer step sees 512 input tokens/511 CE targets, with 1,024 parallel
backbone token evaluations. Checkpointing cuts allocated activation memory
and replays each decoder block; it slows this probe. Host peaks are process
high-water measurements, so later qualification entries inherit prior peaks.
No-checkpointing P=2 fits one T4. The measured budget chooses 500,224 tokens
per arm, not 1M, with a 1.5x training-time safety factor and 1,200-second reserve.
Both 100,352-token smoke and 500,224-token matched checkpoints are saved.

## Full frozen evaluation

| Endpoint | Full-val NLL | Five-domain NLL | LAMBADA NLL | LAMBADA accuracy | HellaSwag accuracy |
| --- | --- | --- | --- | --- | --- |
| Stock | 2.905585 | 2.422360 | 2.258187 | 44.6% | 39.3% |
| Canonical v1 | 2.853786 | 2.384680 | 2.217277 | 45.6% | 40.0% |
| A 500K | 2.864865 | 2.393476 | 2.483144 | 45.6% | 39.1% |
| B 500K | 2.861767 | 2.407465 | 2.587627 | 42.9% | 39.2% |

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| A-stock: full_val_nll | -0.040720 | [-0.043152, -0.038238] |
| B-v1: full_val_nll | +0.007981 | [+0.006302, +0.009653] |
| B-A: full_val_nll | -0.003098 | [-0.004028, -0.002160] |
| B-stock: full_val_nll | -0.043818 | [-0.046242, -0.041300] |

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| B − v1: five_domain_mean_nll | +0.022785 | [+0.019320, +0.026270] |
| B − v1: general_nll | +0.026300 | [+0.016756, +0.036498] |
| B − v1: code_nll | +0.024802 | [+0.015718, +0.034727] |
| B − v1: math_nll | +0.038849 | [+0.032408, +0.045587] |
| B − v1: scientific_nll | +0.012138 | [+0.006877, +0.017698] |
| B − v1: multilingual_nll | +0.011839 | [+0.005728, +0.018137] |
| B − v1: lambada_nll | +0.370350 | [+0.311168, +0.434175] |
| B − v1: lambada_accuracy | -0.027000 | [-0.051000, -0.003000] |
| B − v1: hellaswag_accuracy | -0.008000 | [-0.020000, +0.004000] |

A improves full-val versus stock, but its LAMBADA NLL regresses by +0.224957
(CI [+0.162201,+0.290769]); its observed code and math NLL increases have
intervals spanning zero.
Its research value is a measurable ParScale aggregate gain, not a balanced
release claim. B beats A by a small supported full-val margin, but regresses
against canonical v1 on full-val, every held-out domain, LAMBADA NLL and
LAMBADA accuracy. HellaSwag's B-v1 difference is unresolved.

## PLE contribution and interaction

The trained B checkpoint is unchanged across REAL/PERMUTED/DISABLED inference.
All three contrasts use the pinned 72-block held-out diagnostic.

| B inference-only NLL contrast | Delta | Paired 95% CI |
| --- | --- | --- |
| REAL-DISABLED | -0.022218 | [-0.026163, -0.018351] |
| REAL-PERMUTED | -0.007140 | [-0.009158, -0.005174] |
| PERMUTED-DISABLED | -0.015078 | [-0.017588, -0.012636] |

REAL still beats both controls here. ParScale therefore does not erase the
measurable PLE contribution as global LoRA did; the combined endpoint still
fails the balanced v2 criteria. Pooled support does not imply each small
domain bucket has an individually supported causal gap.

For NLL, gains are stock-v1, stock-A, and stock-B. Interaction is
`gain_combined - (gain_PLE + gain_ParScale) = v1 + A - B - stock`, paired
on the same evaluation blocks.

| Full-val gain / interaction | Gain | Paired 95% CI |
| --- | --- | --- |
| PLE | +0.051799 | [+0.050358, +0.053229] |
| ParScale | +0.040720 | [+0.038238, +0.043152] |
| combined | +0.043818 | [+0.041300, +0.046242] |
| interaction | -0.048701 | [-0.050280, -0.047119] |

The interaction is supported negative, not synergistic. At this 500K endpoint
the methods do not stack into a balanced gain over v1, despite retained PLE
causality. The JSON also retains five-domain and LAMBADA interaction intervals.

## Persistence and stopping

Checkpoint files preserve ParScale weights, AdamW, GradScaler and RNG states
with atomic-save/reopen/hash verification. The private dataset
`ninnix/qwengram-parscale-08b-probe-v2-results/1` was download-back verified.
A-500224 SHA-256 is `03312cc503db94b05766ac7bb91c91f3e1169f334cec417d841353422e18ad3a`;
B-500224 is `3144ac29c7de57a3faa3d67df4a23ba32dacd8ed4d5b4742001539b448722e19`.
They are resumable artifacts, but further ParScale training is not authorized.

The successful inference-only job is
`ninnix/qwengram-parscale-08b-final-eval-fixed-inputs/1`. Startup input attachments
and guards were fixed before relaunch. Native baseline replay maximum block
differences are 1.185e-7 for A and 1.106e-7 for B; trained checkpoint replay is
exact. Frozen tensors remain unchanged, both workers exit zero, and GPU use
stops after evaluation. The job consumes about 20.76 GPU quota minutes.
Results are privately persisted as
`ninnix/qwengram-parscale-08b-final-ablation`; all 36 files are downloaded back
and SHA-256 verified. No automatic retry, continuation or GPU evaluation remains.

Evidence: [full result and per-item arrays](results/parscale.json),
[training protocol](protocols/parscale.json), [full-eval protocol](protocols/parscale-full-eval.json),
[initialization](results/parscale-initialization.json),
[qualification](results/parscale-qualification.json), [budget decision](results/parscale-budget.json),
[resume verification](results/parscale-resumability.json),
[checkpoint receipt](persistence/parscale-checkpoints.json), and
[full-evaluation persistence](persistence/parscale-full-eval.json).
