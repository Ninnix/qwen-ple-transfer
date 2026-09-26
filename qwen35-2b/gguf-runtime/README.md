# Qwengram-2B GGUF runtime validation

Canonical checkpoint: **REAL-10M + linear750**. The [milestone comparison](../milestone-compare/decision.md) retains 15M separately as the max-LM/research endpoint.

| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | PPL reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.538120 | 2.491110 | 0.047010 [0.034128, 0.060324] | 100% | 4.592% |
| Q8_0 | 2.539465 | 2.491114 | 0.048351 [0.035852, 0.061332] | 102.9% [100.9%, 105.9%] | 4.720% |
| Q6_K | 2.551537 | 2.501146 | 0.050391 [0.036843, 0.064777] | 107.2% [100.9%, 114.3%] | 4.91% |
| Q4_K_M | 2.577519 | 2.531151 | 0.046368 [0.032633, 0.060287] | 98.6% [91.8%, 104.5%] | 4.531% |

## Matched test

The original 0.8B runtime protocol is reused: first 64 consecutive 256-token WikiText-2 raw test chunks, scoring the last 127 tokens per chunk (8,128 total). All eight runs use the same CPU build, eight threads, context/batch/microbatch 256, and no warmup. Confidence intervals use 10,000 paired resamples of 16 consecutive four-chunk blocks, seed 1234. Per-chunk values and exact file hashes are in [results.json](results.json); commands and logs are in [logs](logs/).

All 335 backbone tensors and tokenizer fields match exactly within each stock/Qwengram pair. All 11 reader/arbiter tensors remain bit-exact FP32 through BF16, Q8_0, Q6_K and Q4_K_M. See the separate Q6 record below. See [verification](verification.json).

The Q4_1 PLE sidecar is host mapped and has SHA256 `66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`. The [sidecar validation](sidecar-validation.json) checks 4,096 addressed rows against unchanged PyTorch addressing and independent Python Q4_1 dequantization: full prefill, split prefill and single-token decoding all have max absolute difference 0, including EOS boundaries and sequence reset.

This small CPU test uses a Q4_1 PLE. The Kaggle study uses the original FP8 PLE, FP16 backbone execution and different frozen evaluation streams. These scores are separate results, not a numerical reproduction of the Kaggle benchmark.

## Generation checks

On the tested AMD BC-250 Vulkan device, all three 2B precisions with full offload (`-ngl 99`) matched their CPU eight-token greedy continuation for `The capital of France is`, beginning with ` Paris.`. The original 0.8B Q8_0 CPU smoke also passed. The loader rejected missing and invalid PLE sidecars. See [smoke.json](smoke.json). This short check is not a general GPU parity claim.

## Q6_K extension

Q6_K was quantized and tested with fork commit `068fcb42662453bec15298ba6bd59f552190a468`. All 335 backbone tensors and nine tokenizer fields match the stock Q6_K control; all 11 reader and arbiter tensors remain bit-exact FP32. CPU and full Vulkan offload (`-ngl 99`) on the tested AMD BC-250 produced identical eight-token greedy continuations for `The capital of France is`. This short generation check does not establish general GPU parity. See [Q6 verification](q6-verification.json) and [generation outputs](q6-smoke.json).
