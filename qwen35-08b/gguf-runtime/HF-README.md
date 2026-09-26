---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - qwen3.5
  - qwengram
  - gguf
  - safetensors
---

# QwenGram-0.8B, REAL-15M balanced reader

[Research code and results](https://github.com/Ninnix/qwen-ple-transfer) ·
[QwenGram llama.cpp fork](https://github.com/Ninnix/llama.cpp) ·
[Runtime guide](https://github.com/Ninnix/llama.cpp/blob/master/docs/qwengram.md)

This release contains the frozen BF16 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) backbone at revision `2fc06364715b967f1860aea9cf38778875588b17`, the REAL-15M R=1 IDX2 and IDX8 reader, and the separately calibrated 749,568-token linear IDX8 arbiter. It is the balanced canonical candidate in the linked study.

The external PLE is **not included**. The study used the pinned FP8 PLE at `Qwen/Qwen3.8-Flash-Next-FP8` revision `236dfdf285828023ca3bcd3f37366c58a3469b13` (about 48.7 GiB). For local GGUF inference, use the linked llama.cpp fork with Ivan Fioravanti's Q4_1 PLE sidecar below. The fork implements the reader, arbitration, and deterministic n-gram addressing. Unmodified Transformers and llama.cpp do not apply the QwenGram tensors.

**Suggested PLE source for GGUF integrations:** [Ivan Fioravanti's Q4_1 GGUF PLE sidecar](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4/blob/main/Qwen3.8-Flash-Next-PLE-Q4_1.gguf) for Qwen3.8-Flash-Next (32.0 GB). Thank you, Ivan, for making this external PLE available. The reported quality evaluation used the pinned FP8 PLE above; frozen-set quality with Ivan's quantized sidecar has not been measured.

## Run with the llama.cpp fork

Follow the fork's [QwenGram guide](https://github.com/Ninnix/llama.cpp/blob/master/docs/qwengram.md) to build the CPU or Vulkan runtime and obtain the PLE sidecar. For example, from the fork directory after placing both files under `models/qwengram/`:

```sh
export QWENGRAM_PLE="$PWD/models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
build-qwengram-cpu/bin/llama-completion -m models/qwengram/QwenGram-0.8B-Q4_K_M.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

The fork was smoke-tested on CPU with BF16, Q8_0, and Q4_K_M. On the tested AMD BC-250, Q4_K_M selected a different first token with full Vulkan offload; `-ngl 25` matched the CPU first token. These are inference checks, not a replacement for the frozen study evaluation.

## Files

| File | Contents |
| --- | --- |
| `model.safetensors` | Original BF16 Qwen3.5 backbone tensors, plus `qwengram.*` reader and gate tensors in their trained FP32 precision. Original vision and MTP tensors are retained for exact backbone provenance. |
| `QwenGram-0.8B-BF16.gguf` | llama.cpp BF16 text backbone, plus the same `qwengram.*` tensors and QwenGram metadata. No PLE rows. |
| `QwenGram-0.8B-Q8_0.gguf` | Q8_0 quantized text backbone; all 11 reader and gate tensors remain bit-exact FP32. |
| `QwenGram-0.8B-Q6_K.gguf` | Q6_K quantized text backbone; all 11 reader and gate tensors remain bit-exact FP32. |
| `QwenGram-0.8B-Q4_K_M.gguf` | Q4_K_M mixed quantized text backbone; all 11 reader and gate tensors remain bit-exact FP32. |
| `config.json`, tokenizer files, `LICENSE` | Files from the pinned Qwen3.5 base revision. |
| `SHA256.json` | Source identities, file sizes, and SHA-256 hashes. |

The GGUF uses standard `qwen35` backbone tensor names. Extension tensor names mirror the reader state dict: `qwengram.readers.2.*`, `qwengram.readers.8.*`, `qwengram.arbiter.w`, `qwengram.arbiter.b`, and `qwengram.arbiter.alpha2`. The reader uses one key branch and a shared value projection at zero-based decoder layers 2 and 8. The trained gammas are in each reader state dict. Scalar extension tensors are stored as one-element GGUF tensors. The safetensors file retains their original scalar shapes.

The original Q8_0 and Q4_K_M GGUF files were generated directly from the verified BF16 GGUF with llama.cpp `llama-quantize` build 9888 (`cb295bf59`). The command used `--tensor-type '^qwengram[.]=f32'` so the reader and gate stay unchanged. Both files were reopened and checked for the complete tensor inventory, identical QwenGram metadata, and exact extension tensor values. The quantized variants have **not** been separately evaluated on the frozen study sets, so the BF16 metrics below should not be read as quantized-model metrics.

Arbitration uses a frozen `alpha2 = 1.267012596130371` at IDX2 and `alpha8(h8) = 0.5 * sigmoid(dot(w, RMSNorm(h8)) + b)` at IDX8. The norm uses epsilon `1e-6`; `h8` is the pre-injection hidden state. The reader computes in FP32 and casts its residual output to the backbone dtype.

## Frozen evaluation

| Metric | REAL-15M + linear750 |
| --- | ---: |
| Full validation NLL | 2.853786 |
| Full validation perplexity | 17.3534 |
| Five-domain mean NLL | 2.384680 |
| General NLL | 3.134094 |
| Code NLL | 1.495201 |
| Math NLL | 1.402346 |
| Scientific NLL | 2.244491 |
| Multilingual NLL | 3.647268 |
| HellaSwag-1000 accuracy | 0.400 |
| LAMBADA-1000 NLL | 2.217277 |
| LAMBADA-1000 accuracy | 0.456 |

These are the frozen Kaggle study results, not a new evaluation of this export. Frozen-backbone full-validation NLL was 2.905585 (perplexity 18.2759), so the canonical 15M reader reduced perplexity by 5.048%. The 15M reader improved full validation, five-domain mean, and LAMBADA NLL over the 10M reader with paired 95% confidence intervals below zero. Accuracy changes were not established. The study also kept a 20M max-LM endpoint, but 15M remains the balanced choice because 20M regressed on math.

## GGUF runtime retention

A separate [reproducible CPU test](runtime/README.md) scored 8,128 tokens from the first 64 chunks of WikiText-2 raw test. The cached stock BF16 file matches the official pinned Qwen3.5 revision's LFS SHA-256. The original stock Q8_0 and Q4_K_M controls were made with the same quantizer build as the release. Within each stock/QwenGram pair, all 335 backbone tensors and nine tokenizer fields match exactly. The 11 reader and gate tensors remain bit-exact FP32 across the QwenGram GGUFs. QwenGram used Ivan Fioravanti's Q4_1 PLE sidecar. Reader gain is `NLL(stock) - NLL(QwenGram)`, and retention divides the quantized reader gain by the BF16 reader gain.

| Precision | Stock NLL | QwenGram NLL | Reader gain [95% CI] | Gain retention [95% CI] | Perplexity reduction vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.890026 | 2.822039 | 0.067987 [0.058500, 0.077155] | 100% | 6.57% |
| Q8_0 | 2.890255 | 2.822870 | 0.067385 [0.057686, 0.076578] | 99.1% [96.5%, 101.9%] | 6.52% |
| Q6_K | 2.900546 | 2.831509 | 0.069037 [0.059437, 0.078186] | 101.5% [98.3%, 104.7%] | 6.67% |
| Q4_K_M | 2.936757 | 2.874906 | 0.061851 [0.053637, 0.069662] | 91.0% [85.6%, 96.2%] | 6.00% |

Intervals use 10,000 paired resamples of four-chunk blocks and apply to this WikiText slice. Q8_0's reader-gain difference versus BF16 is -0.000602 NLL [-0.002365, 0.001260]; Q4_K_M's is -0.006136 [-0.010367, -0.002394]. These are GGUF runtime measurements, not the frozen Kaggle evaluation or a quality claim for all tasks. Q4_K_M full Vulkan offload gave a different greedy continuation on the tested AMD BC-250; use `-ngl 25` there as noted above. The linked report includes hashes, commands, logs, and per-chunk results.

Q6_K was quantized and tested with fork commit `068fcb42662453bec15298ba6bd59f552190a468`. All 335 backbone tensors and nine tokenizer fields match the stock Q6_K control; all 11 reader and arbiter tensors remain bit-exact FP32. CPU and full Vulkan offload (`-ngl 99`) on the tested AMD BC-250 produced identical eight-token greedy continuations for `The capital of France is`. This short generation check does not establish general GPU parity. See [Q6 verification](runtime/q6-verification.json) and [generation outputs](runtime/q6-smoke.json).

## Provenance

| Artifact | SHA-256 |
| --- | --- |
| REAL-15M reader weights | `e4a760163ec07568178ab48aa533235a9af878183caf120d4e29ce1c8ce4b9dc` |
| REAL-15M linear750 gate | `54b7a98dd5a6b0fcde69deddc8ad2efabfe86daaeca840e949c74ed3bd6d8138` |

Reader training used 15,000,064 tokens; gate calibration used 749,568 tokens. Source reader dataset: `ninnix/qwen-ple-reader-15m-milestone`. Source gate dataset: `ninnix/qwen-ple-reader-scale-15m-arbitration-results`. The original base model and its license are from Qwen. All extension tensors are checked against their published checkpoint hashes before packaging, and both output formats are reopened and tensor-checked before upload.
