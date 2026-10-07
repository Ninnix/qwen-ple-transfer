# LoRA and receiver-local LoRA

## Matched global LoRA

The completed 15M study improves aggregate LM NLL but does not yield a balanced
Qwengram v2. A1 is stock plus LoRA; B1 is canonical v1 plus identical LoRA.
A0/B0 are their frozen baselines. Both training arms receive the same new
token stream, successful-token count, initialization, optimizer and schedule.

Only the 9,338,880 LoRA parameters train: rank 16, alpha 32, dropout zero,
seed 20261003. Qwen3.5 full attention uses q/k/v/o projections (including the
native query output-gate channel); linear attention uses fused `in_proj_qkv`
and `out_proj`; MLPs use gate/up/down. Linear-attention z/a/b projections,
convolution, norms, embeddings, vision and MTP are excluded. Backbone, PLE,
canonical reader, gamma/beta, alpha2 and alpha8 remain frozen.

AdamW peak LR 1e-4, 5% warmup, cosine to zero at 15M, weight decay 0.01,
betas (0.9,0.999), epsilon 1e-8, gradient clip 1; 512 input tokens/step,
batch one, accumulation one. Frozen backbone is FP16; adapters, reader,
arbiter and CE are FP32 with GradScaler. The pinned FineWeb-Edu sample-10BT
stream continues after the complete document containing the 20M reader
endpoint and excludes frozen evaluation/calibration documents and exact spans.
Milestones are 1,000,448, 5,000,192, 10,000,384 and 15,000,064 tokens.

| Endpoint | Full-val NLL | Five-domain NLL | LAMBADA NLL | LAMBADA accuracy | HellaSwag accuracy |
| --- | --- | --- | --- | --- | --- |
| A0 stock | 2.905585 | 2.422360 | 2.258187 | 44.6% | 39.3% |
| B0 v1 | 2.853786 | 2.384680 | 2.217277 | 45.6% | 40.0% |
| A1 15M | 2.810860 | 2.443913 | 2.196620 | 47.8% | 40.9% |
| B1 15M | 2.810854 | 2.440398 | 2.213725 | 46.9% | 40.9% |

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| A1 − stock: full_val_nll | -0.094725 | [-0.097697, -0.091757] |
| A1 − stock: five_domain_mean_nll | +0.021554 | [+0.015698, +0.028482] |
| A1 − stock: lambada_nll | -0.061567 | [-0.104619, -0.017805] |
| B1 − v1: full_val_nll | -0.042932 | [-0.044991, -0.040871] |
| B1 − v1: five_domain_mean_nll | +0.055718 | [+0.050567, +0.061999] |
| B1 − v1: lambada_nll | -0.003551 | [-0.041416, +0.034561] |
| B1 − A1: full_val_nll | -0.000006 | [-0.000324, +0.000319] |
| B1 − A1: five_domain_mean_nll | -0.003515 | [-0.004444, -0.002590] |
| B1 − A1: lambada_nll | +0.017105 | [+0.008213, +0.026286] |

B1 does not establish a full-val improvement over A1, with a delta of only
-0.00000555 NLL and CI spanning zero. B1 has supported regressions on all
five domains versus v1; versus A1, code and LAMBADA NLL regress. It cannot be
promoted on its aggregate gain alone.

The same-checkpoint 72-block PLE diagnostic shows the loss of measurable
REAL-specific memory benefit; no LoRA is retrained between conditions.

| Checkpoint / NLL contrast | Delta | Paired 95% CI |
| --- | --- | --- |
| v1: real-minus-disabled | -0.045320 | [-0.052067, -0.038776] |
| v1: real-minus-permuted | -0.014922 | [-0.017866, -0.011994] |
| v1: permuted-minus-disabled | -0.030398 | [-0.035163, -0.025857] |
| B1 15M: real-minus-disabled | +0.000020 | [-0.003375, +0.003516] |
| B1 15M: real-minus-permuted | +0.002473 | [+0.000737, +0.004289] |
| B1 15M: permuted-minus-disabled | -0.002453 | [-0.004317, -0.000563] |

At B1-15M, REAL does not establish an advantage over DISABLED and is
significantly worse than PERMUTED.
The diagnostic supports a lost REAL-specific contribution at this endpoint;
it does not identify the internal mechanism by which LoRA changes memory use.
The final decision is case 3, no v2 candidate and no automatic later arm.
The study records 22.5171 allocated GPU-hours across its qualified/restarted runs.

## Receiver-local LoRA

This separate test restricts LoRA to the ten eligible projections at the
decoder blocks receiving IDX2/IDX8 memory: 770,048 parameters at zero-based
blocks 2 and 8. It adapts the receiver neighborhood, not the PLE reader.
R-REAL and R-DISABLED have identical adapter shapes/init, optimizer and data
order; only the memory contribution during training differs. Reader, PLE,
canonical arbitration and all base weights stay frozen.

Rank/alpha/dropout/precision and AdamW settings match global LoRA, with a
5M schedule horizon. Data is the exact first 5M prefix of the global study's
15M stream. Milestones are 1,000,448, 2,500,096 and 5,000,192 tokens.
The global 15M arms are contextual references, not token-matched controls.

| Endpoint | Full-val NLL | Five-domain NLL | LAMBADA NLL | LAMBADA accuracy | HellaSwag accuracy |
| --- | --- | --- | --- | --- | --- |
| v1 | 2.853786 | 2.384680 | 2.217277 | 45.6% | 40.0% |
| R-REAL 5M | 2.827969 | 2.402660 | 2.240874 | 46.2% | 40.7% |
| R-DISABLED 5M | 2.828093 | 2.407123 | 2.241884 | 45.4% | 40.8% |

| Comparison / metric | Delta | Paired 95% CI |
| --- | --- | --- |
| R-REAL − v1: full_val_nll | -0.025817 | [-0.026830, -0.024811] |
| R-REAL − v1: five_domain_mean_nll | +0.017981 | [+0.014312, +0.022435] |
| R-REAL − v1: lambada_nll | +0.023598 | [+0.001325, +0.045931] |
| R-REAL − R-DISABLED: full_val_nll | -0.000124 | [-0.000387, +0.000154] |
| R-REAL − R-DISABLED: five_domain_mean_nll | -0.004463 | [-0.005399, -0.003565] |
| R-REAL − R-DISABLED: lambada_nll | -0.001009 | [-0.008175, +0.006158] |

| R-REAL 5M, inference-only NLL contrast | Delta | Paired 95% CI |
| --- | --- | --- |
| real-minus-disabled | -0.004734 | [-0.008468, -0.000930] |
| real-minus-permuted | -0.000319 | [-0.002090, +0.001489] |
| permuted-minus-disabled | -0.004415 | [-0.006672, -0.002097] |

REAL beats DISABLED on the diagnostic, but does not establish a REAL-over-PERMUTED
advantage. Full-val improvement over the matched R-DISABLED arm is unresolved.
The REAL arm regresses versus v1 on general, code, math, scientific and LAMBADA
NLL, and on the five-domain mean. The 5M gate fails; no 10M/15M continuation
occurred. Total recorded allocated GPU-hours are 10.2541.

Evidence: [global analysis](results/lora.json), [receiver decision](results/receiver-lora.json),
[global protocol](protocols/lora.json), [receiver protocol](protocols/receiver-lora.json),
and their [global](persistence/lora.json)/[receiver](persistence/receiver-lora.json)
artifact receipts. The analyses retain intermediate milestones, paired domain
and benchmark arrays, efficiency denominators, adapter hashes and private run
identities. Canonical reader and arbiter identities remain unchanged.
