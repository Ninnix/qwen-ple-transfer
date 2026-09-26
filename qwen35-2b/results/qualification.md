# Qwengram-2B qualification

Pinned before reader training in [`../frozen.json`](../frozen.json): Qwen3.5-2B
revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; target tokenizer
SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`;
Flash-Next PLE source revision `236dfdf285828023ca3bcd3f37366c58a3469b13`.
The target text decoder has hidden size 2048, 24 layers, and vocabulary size
248320. The T4 backbone runs in FP16; the model's linear-attention state and
reader-sensitive reductions use FP32. Three linear-attention blocks precede
each full-attention block. Injection is fixed at IDX2 (human layer 3, 12.5%)
and IDX8 (human layer 9, 37.5%).

The tokenizer check found zero mismatches among shared token IDs, zero
target-only tokens, equal BPE merges, and equal probe encodings. The source
has seven extra audio token IDs beyond the target range. The unchanged
splitmix64 addressing and FP8/sidecar lookup passed the canonical reference
tests. The 0–5M compact cache contains 37,325,743 unique addresses in
5.701 GiB; 2,048 sampled rows matched the immutable master exactly
(maximum absolute difference 0). Its row SHA-256 is
`a355f46fcc7d0ea9b76dee03955acff0948584a1432752ce5ed4856fa3308219`;
its address-map SHA-256 is
`b2a01c076a21f46b3a7b24e386c143a90745291d054a66e7e61fc9eef13559d9`.
The 5M–10M suffix cache contains 37,674,909 unique rows in 5.754 GiB;
its row SHA-256 is
`337b79203d5f74436b23470964c68154ab6cc73ed38f7fc8b3fa30ea0fd9da61`
and address-map SHA-256 is
`350908df637fd8fa4c62ddeed09dcfada54eec4fbf2ed3d2a2f31c18aa2639d7`.
It also matched 2,048 sampled master lookups exactly.
The 10M–15M suffix cache contains 37,711,684 unique rows in 5.760 GiB;
its row SHA-256 is
`7f1cd946387ecf2a8399ded9530353c613a8d0bcef1de9a270e0011fef2bcb0e`
and address-map SHA-256 is
`a197580d91b2eb224e45961d6dc83f78233bc430ffbe59f553ed6534866f3b60`.
It also matched 2,048 sampled master lookups exactly.

The frozen 15,000,064-token FineWebEdu reader stream has SHA-256
`da94e4e3804753b25e4d05b6cef71975e7a943909302c5f3f5d81fddac9b04ef`.
The source dataset revision is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
validation uses the first 549 documents and training reads document indices
549 through 15065. Frozen evaluation and calibration stream SHAs are in
[`evaluation-streams.json`](evaluation-streams.json).

The single-T4 benchmark measured 8.042 GiB training peak VRAM, 4.309 GiB
forward peak VRAM, 810.19 training tokens/s, 3,196.55 forward tokens/s,
4,737 MiB host RSS before the compact cache, and zero allocation on GPU1.
The reader measured 13,059 MiB host RSS at 1M and 13,560 MiB at 5M with the
compact cache mapped, and 7.784 GiB GPU peak. Disabled injection reproduced stock
logits exactly: maximum difference 0, within the declared 0.002 tolerance.
All backbone parameters were frozen and the optimizer contained only the
eight intended reader tensors. Single T4 is the selected configuration.
