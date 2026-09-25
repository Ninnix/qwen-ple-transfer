# AGENT NOTES

## What this is

An experimental research project (not a framework) testing one question: does the pretrained
n-gram PLE of `Qwen/Qwen3.8-Flash-Next-FP8` contain information transferable into a *different*,
frozen Qwen backbone, when only a small target-side reader is trained?

Two independent tracks run the same method on different hardware:

| Track | Target | Compute | Entry |
| --- | --- | --- | --- |
| `qwen36-35b/` | `Qwen/Qwen3.6-35B-A3B`, hidden 2048, 40 layers | Modal A100 80GB | `modal_*.py` + `src/qwen36_ple/` |
| `qwen35-08b/` | `Qwen/Qwen3.5-0.8B`, hidden 1024, 24 layers | Kaggle GPU (recent reader runs: T4 x2) | notebooks, `tools/`, reader study scripts |
| `qwen35-08b/ple-data/` | — | Kaggle CPU | notebook that built the 11 pinned PLE datasets |

## Quality Rules

* Use a minimal programming style.
* Do NOT import new dependencies, import only if strictly necessary; otherwise, start from scratch using the standard 
or already imported libraries.
* Comment with intent, not reflex. Good comments explain function, design, why, teach, guide, or serve as a checklist. 
Skip trivial, debt, and backup comments.

## Commands

Modal (35B). Must run from the repo root: images use `add_local_dir("qwen36-35b/src", ...)`,
a root-relative path. PowerShell needs UTF-8 mode or Modal's status output crashes on Windows:

```powershell
$env:PYTHONUTF8='1'
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_gpu_test.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_prepare.py"          # create/verify the Volume
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_download_ple.py"     # ~48.7 GiB, one time
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_validation_prepare.py"
& ".venv\Scripts\modal.exe" run "qwen36-35b/modal_train.py" --max-tokens 500000 --placement 2
```

`modal_train.py` currently hard-rejects anything but `--max-tokens 500000` and
`--placement 2|13|2-13` (the controlled placement run). The 100K smoke form in the README is
from an earlier revision of that file. Loosen the guard deliberately, not by accident.

Kaggle (0.8B). `verify_notebook.py` is the local test suite — static AST + assertion checks over
notebook cell contents and `kernel-metadata.json`. Run it before every push:

```powershell
python qwen35-08b/verify_notebook.py
python qwen35-08b/tools/push_and_watch.py    # kaggle kernels push -p qwen35-08b, poll, fetch output
```

There is no pytest suite. The gates are the `modal_*.py` entrypoints (each raises on failure and
returns `{"status": "passed", ...}`) and `verify_notebook.py`.

## Architecture

Shared method, identical in both tracks: token ids are hashed to 16 PLE addresses, the FP8 rows
are gathered and dequantized into `e_t in R^2560`, and a reader injects them into the frozen
backbone's residual stream at chosen decoder layers.

```
W_K: 2560 -> hidden   (per branch)     gate = sigmoid(<rms(h), rms(W_K e)>/sqrt(hidden) + beta)
W_V: 2560 -> hidden   (shared)         o    = mean_branches(gate * W_V e)
                                       h    = h + gamma * o        gamma init 0 => exact identity
```

`qwen36-35b/src/qwen36_ple/` is the reference implementation:

- `hashing.py` — `splitmix64` + XOR addressing, prime-sized head tables, and
  `shift_right_ignore_eos` so n-grams never cross a document boundary. **Ground truth**; verified
  bit-exact against the source model. Do not "clean up" the arithmetic.
- `ple_store.py` — `MMapPLEStore.from_manifest`, row gather + `weight_scale` dequant to BF16.
  `tensor_name()` is pinned to source layer 1's shard tensor names.
- `reader.py` / `injection.py` — `SharedValueReader`, and `ReaderInjection` which attaches a
  `register_forward_pre_hook(..., with_kwargs=True)` per site. `set_memory(None)` makes every hook
  a pass-through; that is how the baseline and disabled-memory evaluations are run.
- `config.py` — `ReaderConfig`.

The notebook cells `c-hashing` / `c-reader` / `c-inject` / `c-ple` are a hand port of those four
modules to hidden 1024. Keep them in sync: fix `src/qwen36_ple/` first, then port.

**Layer indexing** is zero-based Transformers index everywhere, called `IDX`. The notebook prints
`HUMAN = IDX + 1` alongside it. Never mix the two in a result table.

### Addressing invariants (never change)

`vocab_size = 248320`, `eos = 248044` (`<|endoftext|>`, the PLE-training terminator — *not* chat
`<|im_end|>` 248046), `seed = 1234`, `ngram = 3`, `heads_per_ngram = 8` (16 slots x 160 dims =
2560), `rows_per_part = 2_500_012`, 128 parts, `vocab_size_base = 20_000_000`. Source and target
BPE id spaces are verified identical. Changing any of these silently invalidates every stored
result and every checkpoint.

### Modal state

One persistent Volume, `qwen36-ple-data`, mounted at `/data`, with `ple/ eval/ checkpoints/
datasets/ hf/ cache/`. Every write is followed by `volume.commit()`. Gate results land in
`/data/eval/*.json`; checkpoints hold reader + optimizer state only, never backbone or PLE.
Each entrypoint is self-contained: its own `modal.App`, its own pinned `pip_install` image, remote
imports inside the function body.

### Invariants the training loop enforces

These checks are the experiment's validity, not defensive noise — preserve them when editing:

- optimizer param ids must equal exactly the reader's trainable param ids;
- no backbone parameter may have a gradient, at any step;
- backbone `_version` counters must not change across a checkpoint;
- PLE file size/mtime must not change mid-run;
- non-finite loss raises immediately;
- HF model/dataset revisions must match those recorded in the frozen validation metadata.

Per SPEC #6, do **not** wrap the target forward in `torch.no_grad()` — gradients must flow through
the frozen layers above the injection. The allowed optimization is `no_grad` on the prefix *below*
the earliest injection site only.

### Validation is a frozen artifact

Both tracks build validation token blocks once, pin them by SHA-256, and abort rather than
overwrite them. Training streams skip the source documents consumed by validation. Baselines are
computed once against that artifact and cached. Placement runs are only comparable because they
share tokens, seed, schedule, and validation SHA — keep new experiments on the same artifact.

### PLE I/O is the bottleneck

Random reads dominate step time. On Modal, cold `mmap` pages fault from the Volume, so warm pages
before measuring throughput. On Kaggle, random reads against `/kaggle/input` cost 97% of the step;
the notebook builds a deduplicated `CompactPLE` cache in one sequential pass (33x faster) and all
sweeps read that. Never add a training path that random-reads `/kaggle/input`.

### Claims discipline

Mandatory controls (SPEC #39): baseline, real memory, disabled memory, permuted addressing,
deterministic random memory. The 0.8B track completed these controls at 500K; the 35B track
still requires its own controls. A validation-loss delta alone is **not** evidence of transfer.
See the track-specific reports before making a transfer claim.

## Local environment

`requirements.txt` pins only the local CLIs (`modal`, `kaggle`); everything else is pinned inside
the Modal images or installed by the notebook.

`qwen35-08b/tools/` is one-off ops: kernel push/poll/fetch runners, log scanners, dataset
forensics. They hardcode absolute Windows paths and scratch log locations and are disposable — do
not refactor them into a library or import from them.
