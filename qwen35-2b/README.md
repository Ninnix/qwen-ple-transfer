# Qwengram-2B transfer track

**Canonical balanced endpoint: REAL-10M + linear750.** The
[matched milestone comparison](milestone-compare/decision.md) retains 10M
under the conservative accuracy rule. It reduces full-val perplexity by
3.473%; 15M remains the stronger LM/research endpoint at 3.700%.
The benchmark accuracy differences are inconclusive; this selection does
not establish statistical superiority of 10M. The results below describe
the completed original 15M study.

**Released:** [BF16, Q8_0, Q6_K and Q4_K_M GGUFs](https://huggingface.co/Ninnix96/Qwengram-2b),
[matched runtime results and PLE sidecar validation](gguf-runtime/README.md),
and [2B support in the llama.cpp fork](https://github.com/Ninnix/llama.cpp/commit/068fcb42662453bec15298ba6bd59f552190a468).
The release preserves both endpoints and all four milestone evaluations.
Large model files remain in the model release and private checkpoint dataset;
this folder contains the study code, notebooks, reports, manifests and evaluation data.
Local Kaggle helpers read authentication from `KAGGLE_API_TOKEN`; the Hugging Face
publisher reads `HF_TOKEN`.

**Completed:** [decision report](results/decision.md),
[full metrics](results/metrics.md), and [artifact definition](qwengram-2b.json).
The final REAL-15M + linear750 candidate reduces full-val perplexity by
3.700% and improves NLL in all five held-out domains. The raw reader reaches
4.047%; arbitration improves LAMBADA while giving up some full-val gain.

The model, tokenizer, PLE addressing and injection placement are pinned in
[`frozen.json`](frozen.json). The 2B text decoder has the same 24-layer count as
the final 0.8B reader, so proportional transfer keeps IDX2 and IDX8 exactly.
The 2B hidden size is 2048. Its alternating three linear-attention and one
full-attention block pattern also matches the 0.8B hook placement semantics:
the reader adds to the residual stream before decoder blocks 2 and 8.

The target tokenizer and pinned Flash-Next tokenizer have identical IDs for
every target token, identical BPE merges, and the reference probe encodes
identically. The seven source-only audio IDs follow the target's highest ID.
The training EOS remains `<|endoftext|>` at ID 248044.

The target weights specify BF16, which a T4 does not support natively. The
frozen backbone must run in FP16; reader and sensitive reductions remain FP32.
Actual memory and speed numbers require a Kaggle T4 run and are not inferred
from the weight size.

The single-T4 qualification benchmark passed: 8.04 GiB peak VRAM, 810
training tokens/s and 3,197 forward tokens/s on 512-token blocks. Disabled
injection had exactly equal logits (max absolute difference 0). GPU1 held no
model allocation. The measured run is in [`results/benchmark-2b.json`](results/benchmark-2b.json),
with the frozen revisions, stream and cache checks in
[`results/qualification.md`](results/qualification.md).

The run is staged at 5M, 10M and 15M tokens so each leg can use a compact
cache for only its new FineWebEdu suffix. `build_train_5m.py` and
`build_train_cont.py` generate the Kaggle notebooks. Each milestone saves the
reader, optimizer, scheduler, RNG, token count and pinned artifact SHAs
atomically. The private Kaggle dataset `ninnix/qwengram-2b-checkpoints`
holds recovery artifacts outside notebook working storage.

The 5M, 10M and 15M milestones completed. Fast validation NLL fell from
2.643400 stock to 2.609224 at 5M and 2.600552 at 10M. At 15M, full-validation
NLL is 2.607793 versus 2.649104 stock (4.047% perplexity reduction). All
three suffix caches reproduced 2,048 sampled master PLE lookups exactly.
Linear750 calibration, final evaluation and the matched controls also completed.

`build_arb_eval.py` generates the frozen-reader linear750 calibration and
evaluation. `build_controls.py` generates the matched 500K controls. These
jobs consumed the same pinned validation and calibration bytes used by the
final 0.8B study. The matched controls establish the expected full-val ordering:
REAL > PERMUTED > RANDOM > DISABLED, with all paired intervals excluding zero.
All 39 entries in the checkpoint/result manifest were downloaded back from
the private Kaggle dataset and verified against their SHA256 and byte size.
