# Qwengram-4B GGUF runtime validation

Canonical checkpoint: **REAL-15M + linear750**. See the [frozen decision](../decision.md) for the independent 10M/15M comparison.

| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | Perplexity reduction vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.225750 | 2.198449 | 0.027301 [0.022650, 0.031917] | 100% | 2.69% |
| Q8_0 | 2.226990 | 2.199469 | 0.027521 [0.022892, 0.032133] | 100.8% [97.1%, 104.7%] | 2.71% |
| Q6_K | 2.230865 | 2.200943 | 0.029922 [0.023478, 0.036439] | 109.6% [99.4%, 118.6%] | 2.95% |
| Q4_K_M | 2.262792 | 2.231828 | 0.030964 [0.025884, 0.036537] | 113.4% [100.6%, 127.6%] | 3.05% |

## Matched test

The matched CPU test scores 8,128 tokens from the first 64 consecutive 256-token WikiText-2 raw test chunks, scoring the last 127 tokens per chunk. All runs use eight threads and context/batch/microbatch 256, with no warmup. Reader gain is `NLL(stock) - NLL(Qwengram)`; retention divides each quantized gain by the BF16 gain. Paired 95% intervals use 10,000 resamples of 16 consecutive four-chunk blocks, seed 1234. The external PLE is Ivan Fioravanti's Q4_1 sidecar.

BF16 has the lowest absolute Qwengram NLL in this test. Gain retention measures the added PLE benefit within each precision.

All eight stock/Qwengram runs were measured with the same patched runtime. Per-chunk scores and model hashes are in [results.json](results.json); commands, timings and logs are in [logs](logs/).

All 441 backbone tensors and nine tokenizer fields match exactly within each stock/Qwengram pair. All 11 reader/arbiter tensors remain bit-exact FP32 across BF16, Q8_0, Q6_K and Q4_K_M and match the verified reader and matching arbiter. See [packaging](packaging.json) and [verification](verification.json).

The [PLE sidecar check](sidecar-validation.json) compares 4,096 addressed rows with independent reference addressing and dequantization. Full prefill, split prefill, token decode, EOS boundaries and repeated reset have max absolute difference zero. The PLE remains unchanged.

This CPU test uses a quantized Q4_1 PLE and a WikiText-2 slice. The frozen Kaggle study uses the original FP8 PLE and different evaluation streams. These scores are separate benchmarks.

## Generation checks

On the tested AMD BC-250, BF16, Q8_0, Q4_K_M with full Vulkan offload (`-ngl 99`) matched the corresponding CPU eight-token greedy continuation for `The capital of France is`. These short checks do not establish broad GPU parity. Q6_K differed at full offload but matched with `-ngl 33`, which keeps the first decoder layer on CPU. Use `-ngl 33` for Q6_K on this tested device, or CPU (`-ngl 0`). The fallback is also a short generation check.

The loader rejected missing and invalid PLE sidecars and conflicting injection indices. The prior 0.8B and 2B Q8_0 CPU continuations remain exact with the updated runtime. See [smoke.json](smoke.json), [Q6 output](q6-smoke.json), [Q6 fallback](q6-fallback.json) and [legacy checks](legacy-smoke.json).

## Runtime source

Published fork: [3616a858f2326e87ad8b48e1341a4e341b3dad73](https://github.com/Ninnix/llama.cpp-qwengram/commit/3616a858f2326e87ad8b48e1341a4e341b3dad73). The tested base commit, source patch and binary hashes are recorded in verification.json; [the source patch](llama-qwengram-4b.patch) reproduces the tested source.
