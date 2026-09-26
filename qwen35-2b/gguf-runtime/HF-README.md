---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
pipeline_tag: text-generation
tags:
- gguf
- qwengram
- external-memory
- llama-cpp
---
# Qwengram-2B

Frozen Qwen3.5-2B plus an R=1 reader at decoder IDX2/IDX8 and linear750 dynamic
arbitration, using external Qwen3.8-Flash-Next PLE memory.

**Canonical balanced endpoint: REAL-10M + linear750** (10,000,384 reader tokens;
749,568 calibration tokens). The matched milestone study found clearer LM gains
at 15M but inconclusive benchmark accuracy differences, including slightly lower
calibrated HellaSwag. We retain 10M under the predeclared conservative selection
rule. This does not establish statistical superiority of 10M.

The canonical checkpoint reduces frozen full-validation perplexity by **3.473%**
(14.141361 stock to 13.650184). The 15M gated research checkpoint reaches **3.700%**
(13.618098), and is preserved under `research-15m/` with its reader and arbiter.
See the [four-arm comparison and paired intervals](evaluation/milestone-decision.md).

## GGUF files

`QwenGram-2B-BF16.gguf`, `QwenGram-2B-Q8_0.gguf`, `QwenGram-2B-Q6_K.gguf` and `QwenGram-2B-Q4_K_M.gguf`
contain the backbone and 11 FP32 reader/arbiter tensors. The reader and arbiter
stay FP32 in every precision. The PLE is a required external file, not embedded
in these GGUFs. `SHA256.json` records all artifact hashes.

## Required PLE sidecar

Download [Ivan Fioravanti's Q4_1 PLE GGUF](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4/blob/main/Qwen3.8-Flash-Next-PLE-Q4_1.gguf).
Credit for this PLE conversion belongs to Ivan. Its SHA256 is
`66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`.
The approximately 32 GB file is mapped on the host; only selected rows are
dequantized for each token. It is not loaded as a 32 GB GPU allocation.

## Build and run

Use the [Qwengram llama.cpp fork](https://github.com/Ninnix/llama.cpp), commit
`068fcb42662453bec15298ba6bd59f552190a468`, which includes 2B support:

```sh
git clone https://github.com/Ninnix/llama.cpp.git
cd llama.cpp
git checkout 068fcb42662453bec15298ba6bd59f552190a468
cmake -S . -B build-qwengram-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-qwengram-cpu -j --target llama-completion
export QWENGRAM_PLE=/path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
build-qwengram-cpu/bin/llama-completion -m /path/to/QwenGram-2B-Q8_0.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

For Vulkan, build with `-DGGML_VULKAN=ON` and use `-ngl 99`. On the tested AMD
BC-250, BF16, Q8_0 and Q4_K_M matched their CPU eight-token greedy smoke output
with full offload. This is a short generation check, not broad GPU parity.

The fork supports both the original 0.8B reader and the 2048-wide 2B reader.
It changes reader dimensions and the corresponding inverse-square-root scale;
injection placement, hashing, PLE lookup and arbitration semantics stay the same.
Stock upstream llama.cpp does not execute this custom reader. MTP and
embedding-only inputs are unsupported for this Qwengram runtime.

## Frozen evaluation

Canonical **REAL-10M + linear750**, using the original FP8 PLE and frozen Kaggle evaluation suite.

| Metric | Frozen stock | Canonical Qwengram-2B |
| --- | ---: | ---: |
| Full-validation NLL | 2.649104 | 2.613753 |
| Full-validation perplexity | 14.141361 | 13.650184 |
| General NLL | 2.841681 | 2.792036 |
| Code NLL | 1.361171 | 1.353190 |
| Math NLL | 1.317180 | 1.306765 |
| Scientific NLL | 2.073600 | 2.050769 |
| Multilingual NLL | 3.309704 | 3.270495 |
| Five-domain mean NLL | 2.180667 | 2.154651 |
| LAMBADA-1000 NLL | 1.825668 | 1.798536 |
| LAMBADA-1000 accuracy | 54.1% | 54.5% |
| HellaSwag-1000 accuracy | 45.6% | 47.0% |

These frozen study metrics are not Q6_K measurements. Quantized GGUF retention is evaluated separately below.

## GGUF runtime retention

The matched CPU test scores 8,128 tokens from the first 64 consecutive 256-token WikiText-2 raw test chunks. All runs use eight threads and context/batch/microbatch 256. Reader gain is `NLL(stock) - NLL(Qwengram)`; retention divides each quantized gain by the BF16 gain. Paired 95% intervals use 10,000 resamples of 16 consecutive four-chunk blocks, seed 1234. The external PLE is Ivan Fioravanti's Q4_1 sidecar.

| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | Perplexity reduction vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.538120 | 2.491110 | 0.047010 [0.034128, 0.060324] | 100% | 4.59% |
| Q8_0 | 2.539465 | 2.491114 | 0.048351 [0.035852, 0.061332] | 102.9% [100.9%, 105.9%] | 4.72% |
| Q6_K | 2.551537 | 2.501146 | 0.050391 [0.036843, 0.064777] | 107.2% [100.9%, 114.3%] | 4.91% |
| Q4_K_M | 2.577519 | 2.531151 | 0.046368 [0.032633, 0.060287] | 98.6% [91.8%, 104.5%] | 4.53% |

Q6_K was quantized and tested with fork commit `068fcb42662453bec15298ba6bd59f552190a468`. All 335 backbone tensors and nine tokenizer fields match the stock Q6_K control; all 11 reader and arbiter tensors remain bit-exact FP32. CPU and full Vulkan offload (`-ngl 99`) on the tested AMD BC-250 produced identical eight-token greedy continuations for `The capital of France is`. This short generation check does not establish general GPU parity. See [Q6 verification](runtime/q6-verification.json) and [generation outputs](runtime/q6-smoke.json).

See the [matched stock-versus-Qwengram runtime report](runtime/README.md).
It uses a fixed WikiText-2 slice and a quantized Q4_1 sidecar; the Kaggle scores
above use the original FP8 PLE and the frozen study suite. Do not treat them as
the same benchmark. The runtime sidecar reproduces reference Q4_1 lookups
exactly for the tested prefill, decode, EOS and reset cases.

The target model revision is `15852e8c16360a2fea060d615a32b45270f8a8fc`.
Full provenance is in `qwengram-2b.json`, the evaluation artifacts, and `runtime/`.
This is an experimental text-generation release; vision has not been validated.
