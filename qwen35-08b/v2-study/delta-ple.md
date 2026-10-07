# Sparse DeltaPLE at 5M

Sparse residual PLE training produces a small, supported full-val gain but
does not pass the strict continuation rule. It remains a separate research
sidecar; canonical v1 and the original PLE are unchanged. The only updated
values are 160-dimensional DeltaPLE rows at addresses actually observed in
training, initialized exactly to zero on first lookup. Evaluation creates
no rows. Backbone, reader, gammas, beta, alpha2 and alpha8 remain frozen.

Training completes 5,000,192 tokens, seed 20261005, 512 tokens/step, SGD
without momentum. FP32 updates/relative L2 feed FP16 CPU storage. Peak LR is
0.01, warmup 5%, cosine to 10% of peak across a predeclared 10M horizon;
the second 5M stage requires the 5M gate and was not launched. Regularization
is `0.1 * mean(||Delta||^2 / max(||PLE||^2,1e-12))` over unique looked-up rows,
with per-update projection to at most 5% of each frozen row's norm, fixed
scale one and gradient clip one.

The modern balanced mixture is FineWeb-Edu 40%, FineWeb2-HQ 15%, Stack-Edu
15%, FineMath-4plus 15%, peS2o v2 10%, Cosmopedia v2 prose 5%. Revisions and
selected shard/content hashes are pinned in
[source records](protocols/delta-ple-sources.json). Documents terminate with
EOS; 512-token blocks are interleaved deterministically. Normalized document
matches and exact 32-token spans against frozen evaluation/calibration are
excluded, and the packed stream is checked again before training. This is
general-LM next-token CE, with no teacher logits or instruction objective.

| Endpoint | Full-val NLL | Five-domain NLL | LAMBADA NLL | LAMBADA accuracy | HellaSwag accuracy |
| --- | --- | --- | --- | --- | --- |
| Canonical v1 | 2.853786 | 2.384680 | 2.217277 | 45.6% | 40.0% |
| DeltaPLE 5M | 2.853629 | 2.384624 | 2.217431 | 45.4% | 40.0% |

| DeltaPLE − v1 metric | Delta | Paired 95% CI |
| --- | --- | --- |
| full_val_nll | -0.000157 | [-0.000175, -0.000140] |
| general_nll | +0.000101 | [+0.000051, +0.000150] |
| code_nll | -0.000250 | [-0.000370, -0.000139] |
| math_nll | +0.000002 | [-0.000082, +0.000084] |
| scientific_nll | -0.000023 | [-0.000088, +0.000039] |
| multilingual_nll | -0.000108 | [-0.000177, -0.000039] |
| lambada_nll | +0.000154 | [-0.000146, +0.000465] |
| lambada_accuracy | -0.002000 | [-0.005000, +0.000000] |
| hellaswag_accuracy | +0.000000 | [+0.000000, +0.000000] |

These table signs invert the NLL gains in the original JSON. The five-domain
mean falls by 0.000055809 NLL. General NLL has a supported regression; code
and multilingual improve. LAMBADA NLL rises by 0.000154082 with CI including
zero, and accuracy drops by two of 1,000 items. The predeclared gate rejects
even these observed LAMBADA regressions; it does not require a supported
regression to stop. No 10M continuation or v2 promotion occurred.

REAL still beats both memory controls on the pooled 72-block diagnostic;
the entire effective PLE is permuted, not just the residual. Each bucket's
mean control-minus-REAL gap is positive, and the pooled lower bounds pass
the >0.01-gap rule. This preserves the combined endpoint's memory benefit;
it does not by itself attribute that benefit to the trained delta.

| Inference-only NLL contrast | Delta | Paired 95% CI |
| --- | --- | --- |
| REAL − PERMUTED | -0.015034 | [-0.018124, -0.012062] |
| REAL − DISABLED | -0.045422 | [-0.052278, -0.038692] |

Every DISABLED per-block score equals frozen v1 exactly. Independent support
verification finds exactly 35,121,420 observed rows, no duplicates or unseen
addresses; 34,841,614 are nonzero. The sidecar uses 11,379,342,389 bytes
(11.379 decimal GB) including metadata. Median relative row norm is 0.005088%,
P95 0.023263%, P99 0.049123%, max 4.999993%. Existing-row training hit fraction
is 52.403963%; held-out hit fraction is 55.676033%.

The full-val gain is only 0.000157353 NLL, or 1.382794e-5 NLL per decimal GB.
This storage cost is a material limitation for a future delta design. The
successful version-4 run took 18,478.94 training seconds and 21,512.04 measured
wall seconds, using one visible T4 on a machine allocating two physical T4s.
A prior filesystem-full failure at 4,454,400 tokens had no complete checkpoint
and restarted from zero. The successful storage layout uses read-only frozen
PLE mappings, the final sidecar as live row storage, and a write-ahead update/
scaler journal. Those failures are not counted as a completed checkpoint.

Evidence: [5M decision](results/delta-ple.json), [protocol](protocols/delta-ple.json),
[stream manifest](protocols/delta-ple-stream.json),
[support verification](results/delta-ple-support.json),
[metric verification](results/delta-ple-metric-verification.json),
[raw evaluation](results/delta-ple/eval-delta-5000192.json),
[raw causal scores](results/delta-ple/causal-delta-5000192.json), and
[persistence receipt](persistence/delta-ple.json). The receipt records verified
download-back sizes/hashes for all 18 artifacts, including the private sidecar.
Its addresses SHA-256 is `0448ca2d5946b4e34212da610c2fc5c49e26ee4fba03cfb7e77b1dfbf0b8a2f9`;
delta payload SHA-256 is `25386c54b4488ca34661c5c75015e7ccbdb1332e98396683087861c73a932f67`.
