# Qwengram-4B

Scaling reproduction of the frozen Qwengram recipe on `Qwen/Qwen3.5-4B`.
The measured study is complete. See [the decision report](decision.md) for the endpoint selection and transfer evidence.

## Canonical release

**REAL-15M + linear750** is the balanced endpoint selected by the frozen rule.
Full-validation perplexity falls from 11.298811 to 10.984210 (**2.784%**).
All five domain NLLs improve over stock and over independently calibrated 10M.
Benchmark accuracy differences remain statistically unresolved.

The subsequent authorized runtime release is
[Ninnix96/Qwengram-4B](https://huggingface.co/Ninnix96/Qwengram-4B), with
**BF16, Q8_0, Q6_K and Q4_K_M** files, the canonical reader and matching arbiter.
All 164 published files passed remote size and SHA256 verification; see the
[publication record](gguf-runtime/publication.json).
See [GGUF runtime validation](gguf-runtime/README.md) for matched stock controls,
quantized gain retention, sidecar checks and CPU/Vulkan generation results.
The [llama.cpp fork](https://github.com/Ninnix/llama.cpp-qwengram) supports
IDX3/IDX11 and keeps all 11 reader/arbiter tensors in FP32. These runtime tests
use the Q4_1 sidecar and are separate from the frozen FP8 study below.

## Frozen setup

| Item | Value |
| --- | --- |
| Backbone revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Decoder | 32 layers, hidden 2560, vocabulary 248320 |
| Early injection | IDX3 / human layer 4 / 12.5% |
| Later injection | IDX11 / human layer 12 / 37.5% |
| Reader | R=1, two sites, FP32 trainable parameters |
| Backbone / PLE | Frozen; backbone FP16 on T4 |
| Reader milestones | 5,000,192 / 10,000,384 / 15,000,064 tokens |
| Arbitration | Separate linear gates for 10M and 15M, 749,568 tokens each |
| Controls | DISABLED / RANDOM / PERMUTED / REAL, 500,224 tokens |

The tokenizer is byte-identical to the 2B tokenizer and preserves the source PLE's
shared token IDs and BPE merges. Both selected 4B sites are full-attention blocks.
These are the frozen relative-depth equivalents; there is no placement sweep.
See [model and tokenizer manifest](frozen.json) and [protocol](protocol.json).

The exact 2B training, calibration and evaluation streams are reused. The original
training manifest retains its 2B source provenance; each 4B checkpoint separately
records the 4B model revision. Training suffix caches must reproduce the existing
2B row and address-map hashes. The mounted master PLE remains read-only.

## Execution

The [single-T4 attempt](https://www.kaggle.com/code/ninnix/qwengram-4b-single-t4-qualification)
passed correctness checks but ran out of memory on its first backward, before any
optimizer step (14.20 GiB allocated, next allocation 486 MiB). See the
[failure record](results/single-t4-failure.json).

At the user's request, the qualified configuration is
[single-T4 activation checkpointing](https://www.kaggle.com/code/ninnix/qwengram-4b-single-t4-checkpointed).
Only GPU0 is visible. The entire FP16 backbone and both FP32 readers remain on that
device. Non-reentrant checkpointing recomputes decoder forward bodies during
backward; injection hooks run once outside those bodies. Backbone evaluation mode,
token blocks, loss, reader math and optimizer schedules are preserved. Qualification
compares logits and all reader gradients against ordinary execution before any
optimizer step, then measures actual memory headroom and throughput.

The [execution manifest](execution.json) freezes this candidate. The earlier
[T4×2 manifest](execution-t4x2.json) is archived; its waiting pipeline monitor was
stopped before changing the active candidate. The checkpointed single-T4 candidate passed qualification on 26 September 2026.
The T4×2 candidate was not needed.

Kaggle's batch jobs initially remained queued, so qualification and the first two
training attempts used the authenticated interactive Jupyter API. Both training
sessions subsequently ended before 5M. Batch slots became available on 26 September,
and the identical qualified notebook was submitted as a batch run (version 3),
resuming the verified 1M recovery checkpoint. The pipeline downloads and verifies
outputs, publishes each milestone to the private checkpoint dataset, and starts
the next stage with the newly persisted checkpoint. This changes execution
transport only; the notebooks and qualification gates are unchanged.

Measured qualification: 406.21 training tokens/s, 1,387.99 forward tokens/s,
9.93 GiB peak allocated VRAM (10.72 GiB reserved), 4,350 MiB host RSS and a
17.48 MiB qualification cache. The full training suffix cache is larger and is
recorded by each training stage. Stock/disabled logits, checkpointed/plain logits
and all eight reader gradient tensors matched exactly in the GPU checks.
See [qualification measurements](results/benchmark-4b.json) and
[execution parity](results/benchmark-execution-parity.json).

```sh
python3 qwen35-4b/build.py
.venv/bin/python qwen35-4b/verify.py
.venv/bin/python qwen35-4b/ops.py pipeline --detach
```

The current batch continuation uses `pipeline --detach`; the completed qualification
is recorded in pipeline state. Only one pipeline may run at a time; a process lock
enforces this.

The first reader session lost its connection after 1.48M confirmed tokens and was
later reported cancelled. Recovery uses the complete 1,000,448-token checkpoint,
verified against a downloaded copy from the private dataset. Reconnection was
tested by deliberately breaking a connection: the cell executed once and its
remote log remained complete. Cells now write a status marker and log on Kaggle;
reconnecting does not submit the current training cell again. See
[recovery provenance](results/recovery-1m.json) and
[transport verification](results/transport-recovery.json).

The second interactive session last reported 2,406,400 tokens at 367.5 tokens/s
before Kaggle ended it (operation error code 1; Jupyter HTTP 404). Its last durable
checkpoint remained 1,000,448 tokens. This progress is historical, not a completed
milestone. See [second interruption](results/interrupted-run-353083069.json).

The pipeline waits for qualification, then runs 0–5M, 5–10M and 10–15M reader
training, independent 10M/15M arbitration and evaluation, and the matched controls.
It stops on failed qualification or a failed notebook. It verifies checkpoint
contents and SHAs, publishes each milestone to the private Kaggle checkpoint
dataset, and verifies a downloaded copy before launching the next stage.
Local large artifacts and operational logs live in the ignored `.artifacts/` folder.
Set `KAGGLE_API_TOKEN` for CLI access.

The original reference schedule is preserved: the internal 1M recovery checkpoint
is followed by the established continuation schedule through 5M, 10M and 15M.
Neither 5M nor 10M is a stopping decision based on diminishing gains.

## Monitoring

The local user timer `qwengram-4b-monitor.timer` was stopped after the final report
was persisted and verified. During training it queued a check in the existing
chat every 30 minutes. Each check inspected current Kaggle activity, training
progress, persisted checkpoints and failures. At most one check remains queued
while the chat is unavailable. The computer and Codex service must
remain available. Delivery records are in `results/monitor.json`; operational
history is in the ignored artifact directory. Monitoring stops after the final
report is persisted. To stop it manually:

```sh
systemctl --user stop qwengram-4b-monitor.timer
```

## Endpoint decision

Report raw and calibrated 15M-minus-10M paired 95% intervals for full validation,
all five domains, domain mean, LAMBADA NLL/accuracy, and HellaSwag accuracy.
Use 10,000 paired resamples with seed 1234 and the reference bootstrap logic.

Choose calibrated 15M only if both aggregate NLL intervals are entirely below zero,
with no individual domain/LAMBADA NLL interval entirely above zero and no benchmark
accuracy interval entirely below zero. Otherwise prefer calibrated 10M and retain
15M as the research endpoint. Preserve both raw readers. An unresolved regression
does not establish equivalence.

The final [decision.md](decision.md) includes all measured endpoints, transfer controls
and the cross-scale comparison with 0.8B and the 2B canonical decision frozen when
this study began. The study ended at that report. The user subsequently authorized
the GGUF/runtime release documented above; no further training or architecture
experiment was added.
