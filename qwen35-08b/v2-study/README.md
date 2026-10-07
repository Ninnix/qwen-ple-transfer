# Qwengram-0.8B follow-up studies toward v2

These are completed research studies, recorded on October 7, 2026. **Qwengram
v1 remains canonical.** No endpoint here passed the balanced promotion rules.
The record preserves useful improvements, failed combinations, causal controls,
and checkpoint identities for future research. It changes documentation and
result records only.

| Study | Actual new training | Updated parameters | Decision |
| --- | --- | --- | --- |
| [Matched global LoRA](lora.md) | 15,000,064 tokens per arm | 9,338,880 LoRA parameters | Full-val improves; domains regress and REAL-specific PLE benefit is no longer measurable |
| [Receiver-local LoRA](lora.md#receiver-local-lora) | 5,000,192 tokens per arm | 770,048 LoRA parameters at IDX2/IDX8 | No-go at 5M; no 10M continuation |
| [PLE reader and routing](reader.md) | 749,568 head-router tokens; separate utility-label gate study | 16,400 head-router or 1,025 late-gate parameters | Raw routing gain depends on amplitude; utility gate regresses on LAMBADA |
| [Sparse DeltaPLE](delta-ple.md) | 5,000,192 tokens | Observed 160-dimensional residual PLE rows | Tiny gain, 11.38 GB sidecar, strict LAMBADA gate fails |
| [Official-style ParScale P=2](parscale.md) | 500,224 tokens per arm | 9,767,938 prefix/aggregation parameters | A is a ParScale-only research endpoint; B is a negative combination |

The preceding [R=1 reader scaling](../reader-scale/results/decision.md),
[15M canonical endpoint](../reader-scale-15m/results/decision.md),
[20M endpoint](../reader-scale-20m/results/decision.md), and
[R=4 warm start](../reader-r4/results/decision.md) remain in their existing reports.
The 20M/R=4 aggregate gains did not preserve the domain balance of 15M R=1.

## Frozen comparison

| Endpoint | Full-val NLL | Five-domain NLL | LAMBADA NLL | LAMBADA accuracy | HellaSwag accuracy |
| --- | --- | --- | --- | --- | --- |
| Stock | 2.905585 | 2.422360 | 2.258187 | 44.6% | 39.3% |
| Canonical v1 | 2.853786 | 2.384680 | 2.217277 | 45.6% | 40.0% |
| Stock + LoRA, 15M | 2.810860 | 2.443913 | 2.196620 | 47.8% | 40.9% |
| v1 + LoRA, 15M | 2.810854 | 2.440398 | 2.213725 | 46.9% | 40.9% |
| Receiver LoRA REAL, 5M | 2.827969 | 2.402660 | 2.240874 | 46.2% | 40.7% |
| Receiver LoRA DISABLED, 5M | 2.828093 | 2.407123 | 2.241884 | 45.4% | 40.8% |
| Head router, 750K | 2.851222 | 2.381964 | 2.216285 | 45.6% | 40.1% |
| Same head router, norm-matched | 2.853946 | 2.384701 | 2.216906 | 45.5% | 39.9% |
| Utility-trained late gate | 2.851418 | 2.383219 | 2.230427 | 45.4% | 40.1% |
| Sparse DeltaPLE, 5M | 2.853629 | 2.384624 | 2.217431 | 45.4% | 40.0% |
| ParScale A, 500K | 2.864865 | 2.393476 | 2.483144 | 45.6% | 39.1% |
| ParScale B, 500K | 2.861767 | 2.407465 | 2.587627 | 42.9% | 39.2% |

Budgets are matched **within** each study, not across studies. The LoRA studies
use the pinned FineWeb-Edu continuation; DeltaPLE and ParScale use a modern
balanced stream; head routing uses canonical calibration. B arms inherit v1's
15,000,064 reader and 749,568 arbitration tokens. New-token matching does not
match total historical compute. The utility-gate means above are reconstructed
from canonical means plus the saved paired deltas.

The suite is the unchanged 1,024 x 512-token validation blocks, 64 x 512-token
blocks in each of general/code/math/scientific/multilingual, HellaSwag-1000,
and LAMBADA-1000 accuracy/NLL. The five-domain mean weights domains equally.
Reported intervals are paired percentile bootstraps with 10,000 resamples,
unadjusted 95%; blocks for domain/validation NLL and items for benchmarks.
Five-domain intervals resample each domain separately. Seeds are preserved per
study: 1234 for LoRA, utility and ParScale; 20261006 for head/norm routing.
DeltaPLE's original intervals and executed analysis identity are retained.

Report tables use candidate minus reference: negative NLL is better, positive
accuracy is better. Accuracy deltas are fractions unless marked percentage
points. DeltaPLE's original JSON stores NLL *gains* in the opposite direction;
its report explicitly converts them. CIs including zero do not establish
equivalence. These intervals quantify uncertainty on the fixed evaluation
sample, not training-seed variance. No multiple-comparison adjustment is claimed.
The 72-block memory diagnostic is a held-out subset, not full-suite causal
evaluation: 32 validation blocks plus eight per domain, permutation seed 777,
same trained checkpoint for REAL/PERMUTED/DISABLED.

## Canonical identities

| Artifact | Pinned identity |
| --- | --- |
| `Ninnix96/Qwengram-0.8B` release | `9efeb01d97b8156062e0a524a70b417dd7f045a7` |
| `Qwen/Qwen3.5-0.8B` backbone | `2fc06364715b967f1860aea9cf38778875588b17` |
| `Qwen/Qwen3.8-Flash-Next-FP8` PLE source | `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| Canonical reader SHA-256 | `e4a760163ec07568178ab48aa533235a9af878183caf120d4e29ce1c8ce4b9dc` |
| Canonical arbiter SHA-256 | `54b7a98dd5a6b0fcde69deddc8ad2efabfe86daaeca840e949c74ed3bd6d8138` |

v1 uses R=1 with shared value projection, canonical decoder-input IDX2/IDX8
injections (human layers 3/9), alpha2=1.267012596130371, and the frozen dynamic
alpha8 arbiter. Addressing remains seed 1234, vocab 248320, EOS 248044, and
16 addressed rows of 160 values (eight bigram plus eight trigram heads).
Each report states its trainable scope. Historical utility training creates
a separate late-gate endpoint; it does not overwrite canonical arbitration.

## Evidence and continuation

[provenance.json](provenance.json) lists the original path, byte size and SHA-256
of every copied evidence file. [results/summary.json](results/summary.json)
provides the endpoint means and executed budgets. The seven main result JSONs
retain the original paired CIs, decisions, numerical/freeze audits and artifact
references. LoRA, receiver-LoRA and ParScale JSONs include per-block/item
evaluation arrays; reader-routing and DeltaPLE arrays are in their result
subdirectories. Utility retains its paired analysis and private artifact receipt.
Large checkpoints, PLE/cache bytes, training/evaluation text and credentials
are excluded from this Git record. Private artifact references require access
to the original account; hashes permit verification without exposing keys.

Protocols are byte-preserved historical plans. Fields such as `status`,
`actual_tokens`, proposed milestones, `publish=false` and optional continuations
may describe preparation or model-release restrictions. The completed decisions
and actual budgets in this report take precedence: DeltaPLE stopped at 5M,
receiver LoRA stopped at 5M, and ParScale stopped at matched 500K. Publishing
this research record does not promote or publish any experimental model.

Future improvements should start from canonical v1 and preserve the frozen
suite. Three untested directions follow from this evidence: constrain injection
amplitude during router *training*, improve utility-label calibration while
checking LAMBADA, and seek a much smaller DeltaPLE parameterization. They are
hypotheses, not established gains or launched experiments. LoRA's loss of
REAL-specific benefit and ParScale's benchmark/domain regressions must remain
explicit acceptance checks. No further ParScale training, P=4 run, or GPU
evaluation is scheduled; the 500K full evaluation is its final ablation.
