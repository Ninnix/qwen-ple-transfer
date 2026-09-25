# QwenGram GGUF runtime validation

This test compares the [REAL-15M QwenGram GGUFs](https://huggingface.co/Ninnix96/Qwengram-0.8B) with the same frozen Qwen3.5-0.8B backbone at each precision. It uses the [QwenGram llama.cpp fork](https://github.com/Ninnix/llama.cpp) at commit `1c5053f23e99341bee106c1232af90f57ed2cfdd`.

## Reader gain and retention

For each precision, reader gain is `NLL(stock) - NLL(QwenGram)`. Retention is the quantized reader gain divided by the BF16 reader gain. NLL is in nats per scored token; lower is better.

| Precision | Stock NLL | QwenGram NLL | Reader gain [95% CI] | Retention [95% CI] | Perplexity reduction vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.890026 | 2.822039 | 0.067987 [0.058500, 0.077155] | 100% | 6.57% |
| Q8_0 | 2.890255 | 2.822870 | 0.067385 [0.057686, 0.076578] | 99.1% [96.5%, 101.9%] | 6.52% |
| Q4_K_M | 2.936757 | 2.874906 | 0.061851 [0.053637, 0.069662] | 91.0% [85.6%, 96.2%] | 6.00% |

Q8_0's gain loss versus BF16 is -0.000602 NLL, with paired interval [-0.002365, 0.001260]. Q4_K_M's is -0.006136 [-0.010367, -0.002394]. These intervals describe this WikiText slice; they are not the frozen study's intervals.

## Inputs and checks

- The stock BF16 source is `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`. Its downloaded safetensors SHA-256, `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, matches the official Hugging Face LFS SHA.
- The stock BF16 GGUF was converted from that source. Stock Q8_0 and Q4_K_M were made with the same `llama-quantize` build 9888 (`cb295bf59`) as the released QwenGram quants. All 335 backbone tensors and all nine tokenizer fields match exactly between each stock/QwenGram pair, including the quantized tensor types.
- The 11 QwenGram reader and gate tensors match the SHA-verified 15M checkpoints in BF16 GGUF. They remain bit-exact FP32 in Q8_0 and Q4_K_M, and QwenGram metadata is unchanged. The GGUFs contain no PLE rows.
- QwenGram runs used [Ivan Fioravanti's Q4_1 PLE GGUF](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4/blob/main/Qwen3.8-Flash-Next-PLE-Q4_1.gguf), SHA-256 `66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`. The runtime rejects the model when `QWENGRAM_PLE` is unset.

The input is the first 64 consecutive 256-token chunks of [WikiText-2 raw test](https://huggingface.co/datasets/ggml-org/ci/blob/main/wikitext-2-raw-v1.zip), file SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. `llama-perplexity` scores the last 127 tokens of each chunk: 8,128 scored tokens total. All six models ran on CPU with eight threads, context and batch size 256, and no warmup. The intervals use 10,000 paired resamples of 16 consecutive four-chunk blocks, seed 1234. [Run logs](logs/) and [per-chunk results](results.json) are preserved; run `python analyze.py` here to recalculate.

For each model, the command was:

```sh
QWENGRAM_PLE=/path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  llama-perplexity -m /path/to/model.gguf -f wiki.test.raw \
  -c 256 -b 256 -ub 256 --chunks 64 --ppl-output-type 1 \
  --no-warmup -t 8 --log-file /path/to/log
```

Stock runs omit `QWENGRAM_PLE`. The control GGUFs and all input/model hashes are listed in [results.json](results.json).

## Runtime scope

CPU greedy generation of `The capital of France is` returned ` Paris.` for BF16, Q8_0, and Q4_K_M. On the tested AMD BC-250 Vulkan device, BF16 and Q8_0 with `-ngl 99` and Q4_K_M with `-ngl 25` returned ` Paris.`; Q4_K_M with `-ngl 99` generated a different continuation. Use `-ngl 25` for Q4_K_M on this device. This generation check is separate from the CPU NLL table.

This GGUF test uses a quantized PLE sidecar and a small WikiText-2 slice. The original Kaggle study used the pinned FP8 PLE and different frozen evaluation sets. Its canonical 15M full-validation NLL was 2.853786 versus 2.905585 for the frozen backbone: perplexity 17.3534 versus 18.2759, a 5.048% reduction. The GGUF numbers above are not a reproduction of that benchmark.
