# qwen36-ple

Target-side reader adaptation for transferring the frozen n-gram PLE from
`Qwen/Qwen3.8-Flash-Next-FP8` to the frozen text backbone of
`Qwen/Qwen3.6-35B-A3B` on one Modal A100 80GB.

The implementation follows `SPEC.md`. The current result is a completed 100K-token
pipeline smoke test, not evidence that the transferred memory improves the target.

## Current status

Passed:

- Modal A100 80GB BF16 smoke test.
- Persistent `qwen36-ple-data` volume.
- Text-only Qwen3.6 bitsandbytes INT8 loading with no missing or unexpected weights.
- Frozen-backbone backward proof and exact `gamma=0` identity.
- Target/source tokenizer semantic equality.
- Exact Qwen4-Exp n-gram hash equivalence.
- Real FP8 PLE row and dequantization equivalence.
- Full-table mmap I/O benchmark.
- Integrated 512-token real-PLE forward.
- 100K-token FineWeb-Edu reader training smoke.

Completed placement comparison:

- 500K-token placement ablations at layers 2, 13, and 2+13.

Not yet run:

- Random, permuted-address, and disabled-memory controls.
- 5M-token transfer experiment.

## Measured results

### Target

- GPU: NVIDIA A100 80GB PCIe.
- Text model: `Qwen3_5MoeForCausalLM`.
- Quantization: bitsandbytes INT8.
- Model allocation after load: 63.251 GiB.
- 512-token inference peak: 64.089 GiB.
- Warm 512-token inference: 642.02 tokens/s.

### Reader proof

- Placement: target layer 2, zero-based Transformers layer index.
- Branches: 1.
- Trainable parameters: 10,485,762.
- Identity maximum absolute logit difference: 0.0.
- Backbone gradients: 0.
- Peak allocation for a 32-token optimizer step: 65.209 GiB.

### Source PLE

- Source architecture: Qwen4-Exp.
- Source revision: `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- PLE checkpoint module: zero-based source layer 1, one-indexed source layer ID 2.
- Hash PLE ordinal: 0.
- Logical table: 128 parts of `[2,500,012, 160]` FP8 E4M3 rows.
- Concatenated memory per token: 16 x 160 = 2560 dimensions.
- Physical files: source checkpoint shards 5 through 37 only.
- Downloaded PLE size: 48.671 GiB.
- BF16 scale: `0.00019931793212890625`.
- Real-row reference maximum absolute difference: 0.0.
- Sustained distinct-address lookup after startup faults: 3,020-5,908 tokens/s.

Cold mmap lookup faults pages from the Modal Volume and is much slower. Long-lived jobs
must warm representative PLE pages before measuring throughput.

### Integrated forward

- Sequence length: 512.
- Real-PLE identity maximum absolute difference: 0.0.
- Augmented forward: 429.02 tokens/s.
- Peak allocation: 64.583 GiB.

### 100K training smoke

- Corpus: `HuggingFaceFW/fineweb-edu`, `sample-10BT`.
- Sequence length: 512.
- Steps: 196.
- Tokens: exactly 100,000.
- Throughput: 174.89 tokens/s.
- Peak allocation: 74.489 GiB.
- Mean training loss: 2.2491.
- Frozen baseline validation loss: 2.2578.
- Initial reader validation loss: 2.2617.
- Final reader validation loss: 2.2617.
- Final gamma: 0.0050354.

The smoke proves correctness and stability, but it does not show transfer benefit. The
validation set is intentionally tiny and the final loss is slightly worse than baseline.
Placement ablations and mandatory controls are required before a GO decision.

### Fixed validation and placement

All placement runs use the same frozen validation artifact:

- Tokens: 524,288.
- Sequences: 1,024 x 512.
- SHA-256: `451a8ec49c1f197db1cf77839d8304a5512dfe954f6722bccbff86f9ac49477c`.
- FineWeb-Edu revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- Training skips the first 549 source documents used to construct validation.
- Frozen Qwen baseline loss: 2.21511767.

Each 500K experiment used 978 optimizer steps, 49 warmup steps (5.01%), sequence
length 512, seed 1234, and checkpoints at 100K, 250K, and 500K tokens.

Final placement results:

| Placement | Validation loss | Delta vs baseline | Gamma | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| layer 2 | 2.21427982 | -0.00083785 | 0.008911 | 74.917 GiB |
| layer 13 | 2.21506216 | -0.00005551 | 0.007996 | 72.081 GiB |
| layers 2+13 | 2.21471746 | -0.00040021 | 0.007996 each | 75.008 GiB |

Layer 2 improved monotonically across checkpoints:

| Tokens | Validation delta | Gate mean | Gate std | W_K update | W_V update |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | -0.00000619 | 0.6405 | 0.1971 | 0.6346 | 0.9078 |
| 250K | -0.00031913 | 0.7819 | 0.1672 | 1.0522 | 1.8091 |
| 500K | -0.00083785 | 0.7919 | 0.1634 | 1.0870 | 1.8923 |

The reader is updating and produces a small repeatable validation improvement. Layer 2
is the selected placement. The result is not yet evidence that pretrained PLE content is
responsible: disabled, permuted-address, and deterministic random-memory controls remain
mandatory before a 5M run.

## Artifacts

The Modal Volume contains:

```text
/data/ple/qwen3.8-flash-next-fp8/manifest.json
/data/eval/source_ple_metadata.json
/data/eval/target_load.json
/data/eval/reader_proof.json
/data/eval/ple_row_test.json
/data/eval/ple_io.json
/data/eval/integrated_forward.json
/data/eval/fixed_validation_baseline.json
/data/eval/layer2-r1-500k.json
/data/eval/layer13-r1-500k.json
/data/eval/layer2-13-r1-500k.json
/data/checkpoints/smoke-100000-layer2-r1/
/data/checkpoints/layer2-r1-500k/
/data/checkpoints/layer13-r1-500k/
/data/checkpoints/layer2-13-r1-500k/
```

The checkpoint contains only reader state, optimizer state, metrics, and metadata. It
does not duplicate Qwen3.6 or the PLE.

## Commands

PowerShell must use UTF-8 mode for Modal's status output on Windows:

```powershell
$env:PYTHONUTF8='1'
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_gpu_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_prepare.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_target_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_reader_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_source_inspect.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_hash_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_ple_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_ple_benchmark.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_integrated_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_train.py" --max-tokens 100000
```

`modal_train.py` rejects a missing or non-positive token budget.
