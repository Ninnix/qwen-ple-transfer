---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
tags:
- gguf
- qwengram
- external-memory
- llama-cpp
---
# Qwengram-4B

Frozen Qwen3.5-4B plus an R=1 reader at decoder **IDX3/IDX11** (human layers
4/12, 12.5%/37.5% of 32 layers), with a learned global early multiplier and
linear token-dependent later arbitration. The external Qwen3.8-Flash-Next PLE
supplies the memory. The backbone and PLE remain frozen.

The early multiplier is a learned scalar bounded to [0.75, 1.75]. The later
multiplier is `0.5 * sigmoid(w @ RMSNorm(h_late) + b)`, computed per token.

**Canonical endpoint: REAL-15M + linear750** (15,000,064 reader tokens;
749,568 calibration tokens). The matched 10M/15M study independently calibrated
both readers. Calibrated 15M improves full-validation, all five domains and
their mean over calibrated 10M with paired 95% NLL intervals below zero,
without a statistically clear benchmark regression. Unresolved accuracy
differences do not establish equivalence.

Frozen full-validation perplexity falls **2.784%**, from **11.298811 to
10.984210**. Raw 15M achieves a larger 2.970% reduction; calibration trades
some aggregate LM gain for lower LAMBADA NLL. Matched 500K controls resolve
REAL > PERMUTED > RANDOM > DISABLED. See the [decision report](evaluation/decision.md)
for all four endpoints, paired intervals, controls and cross-scale provenance.

## GGUF files

| File | Backbone precision |
| --- | --- |
| [QwenGram-4B-BF16.gguf](QwenGram-4B-BF16.gguf) | BF16 |
| [QwenGram-4B-Q8_0.gguf](QwenGram-4B-Q8_0.gguf) | Q8_0 |
| [QwenGram-4B-Q6_K.gguf](QwenGram-4B-Q6_K.gguf) | Q6_K |
| [QwenGram-4B-Q4_K_M.gguf](QwenGram-4B-Q4_K_M.gguf) | Q4_K_M |

All files contain the backbone plus **11 FP32 reader/arbiter tensors**. These
extension tensors stay FP32 in every precision. `reader.safetensors` and
`arbiter.pt` contain the same canonical checkpoint separately. The PLE is a
required external file. [SHA256.json](SHA256.json) records hashes and sizes.

## Required PLE sidecar

Download [Ivan Fioravanti's Q4_1 PLE GGUF](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4/blob/main/Qwen3.8-Flash-Next-PLE-Q4_1.gguf).
Credit for this PLE conversion belongs to Ivan. Its SHA256 is
`66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`.
The approximately 32 GB file is mapped on the host; selected rows are
dequantized per token. It does not require a 32 GB GPU allocation.

## Build and run

Use the [Qwengram llama.cpp fork](https://github.com/Ninnix/llama.cpp-qwengram) at commit `3616a858f2326e87ad8b48e1341a4e341b3dad73`:

```sh
git clone https://github.com/Ninnix/llama.cpp-qwengram.git
cd llama.cpp-qwengram
git checkout 3616a858f2326e87ad8b48e1341a4e341b3dad73
cmake -S . -B build-qwengram-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-qwengram-cpu -j --target llama-completion
export QWENGRAM_PLE=/path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
build-qwengram-cpu/bin/llama-completion -m /path/to/QwenGram-4B-Q8_0.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

For Vulkan, build with `-DGGML_VULKAN=ON`. On the tested AMD BC-250, BF16, Q8_0, Q4_K_M with full Vulkan offload (`-ngl 99`) matched the corresponding CPU eight-token greedy continuation for `The capital of France is`. These short checks do not establish broad GPU parity. Q6_K differed at full offload but matched with `-ngl 33`, which keeps the first decoder layer on CPU. Use `-ngl 33` for Q6_K on this tested device, or CPU (`-ngl 0`). The fallback is also a short generation check.

The fork supports the original 0.8B/2B readers at IDX2/IDX8 and the 2560-wide
4B reader at IDX3/IDX11. Hashing, row ordering and reader math are preserved.
Stock upstream llama.cpp does not execute this custom reader. MTP and
embedding-only inputs are unsupported. Vision has not been validated.

## Frozen evaluation

Canonical **REAL-15M + linear750**, using the original FP8 PLE and frozen Kaggle suite.

| Metric | Frozen stock | Canonical Qwengram-4B |
| --- | ---: | ---: |
| Full-validation NLL | 2.424697 | 2.396459 |
| Full-validation perplexity | 11.298811 | 10.984210 |
| General NLL | 2.530256 | 2.498815 |
| Code NLL | 1.221217 | 1.204523 |
| Math NLL | 1.212598 | 1.196319 |
| Scientific NLL | 1.894821 | 1.891064 |
| Multilingual NLL | 2.975046 | 2.950582 |
| Five-domain mean NLL | 1.966787 | 1.948261 |
| LAMBADA-1000 NLL | 1.350000 | 1.336395 |
| LAMBADA-1000 accuracy | 65.9% | 66.6% |
| HellaSwag-1000 accuracy | 54.8% | 55.1% |

All five domains improve over stock with paired NLL intervals below zero.
LAMBADA NLL improves significantly; HellaSwag and LAMBADA accuracy gains are
point estimates with intervals overlapping zero. Training used a single T4
with decoder activation checkpointing, with a 10.029 GiB measured peak.
The [study](https://github.com/Ninnix/qwen-ple-transfer/tree/main/qwen35-4b)
and [evaluation artifacts](evaluation/) preserve the protocol, per-item scores,
paired bootstrap results and checkpoint identities. The historical cross-scale
table retains the 2B canonical choice frozen when that study began.

## GGUF runtime retention

The matched CPU test scores 8,128 tokens from the first 64 consecutive 256-token WikiText-2 raw test chunks, scoring the last 127 tokens per chunk. All runs use eight threads and context/batch/microbatch 256, with no warmup. Reader gain is `NLL(stock) - NLL(Qwengram)`; retention divides each quantized gain by the BF16 gain. Paired 95% intervals use 10,000 resamples of 16 consecutive four-chunk blocks, seed 1234. The external PLE is Ivan Fioravanti's Q4_1 sidecar.

| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | Perplexity reduction vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.225750 | 2.198449 | 0.027301 [0.022650, 0.031917] | 100% | 2.69% |
| Q8_0 | 2.226990 | 2.199469 | 0.027521 [0.022892, 0.032133] | 100.8% [97.1%, 104.7%] | 2.71% |
| Q6_K | 2.230865 | 2.200943 | 0.029922 [0.023478, 0.036439] | 109.6% [99.4%, 118.6%] | 2.95% |
| Q4_K_M | 2.262792 | 2.231828 | 0.030964 [0.025884, 0.036537] | 113.4% [100.6%, 127.6%] | 3.05% |

BF16 has the lowest absolute Qwengram NLL in this test. Gain retention measures
the added PLE benefit within each precision.

All 441 backbone tensors and nine tokenizer fields match the corresponding
stock controls; all 11 extension tensors remain bit-exact FP32. See
[tensor verification](runtime/verification.json), [sidecar validation](runtime/sidecar-validation.json),
[generation checks](runtime/smoke.json) and the [matched report](runtime/README.md).
The Q4_1 runtime test and the FP8 frozen evaluation above are separate benchmarks.

The target model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Full provenance is in [qwengram-4b.json](qwengram-4b.json), `evaluation/`, and `runtime/`.
Study-relative paths in the checkpoint definition resolve in the linked study repository.
This is an experimental text-generation release.
