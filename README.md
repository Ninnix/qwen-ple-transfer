# qwen-ple-transfer

Cross-model transfer of the frozen n-gram PLE (Engram memory) from
`Qwen/Qwen3.8-Flash-Next-FP8` into frozen Qwen text backbones by training
only a small target-side reader. Backbone and PLE stay frozen; only reader
parameters receive optimizer updates.

Shared method: shared-value reader projections with a contextual sigmoid gate
and residual injection (`h = h + gamma * o`), identity init (`gamma = 0`),
exact source addressing (`splitmix64-xor`, seed 1234), standard causal-LM loss,
mandatory disabled/permuted/random-memory controls.

## Getting started

Use Python 3.11 or newer from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python qwen35-08b/verify_notebook.py
python qwen35-08b/tools/check_builder.py
```

`requirements.txt` pins the local Kaggle and Modal command line tools. The
Kaggle notebooks install their GPU runtime dependencies. These local checks
do not launch training.

Use newly issued credentials: the pre-cleanup private Git history contained
a Hugging Face token and an older Kaggle token. Revoke or rotate those tokens
before sharing this repository. Keep credential values in your shell or
account secret manager. In Bash, these prompts avoid putting values in shell
history:

```sh
read -rsp 'HF_TOKEN: ' HF_TOKEN; printf '\n'; export HF_TOKEN
read -rsp 'KAGGLE_API_TOKEN: ' KAGGLE_API_TOKEN; printf '\n'; export KAGGLE_API_TOKEN
python qwen35-08b/tools/kaggle_check.py
```

`HF_TOKEN` needs access to the pinned Qwen models. `KAGGLE_API_TOKEN` is used
by local Kaggle CLI commands. Kaggle does not copy local shell variables into
a remote notebook. Attach account secrets named `HF_TOKEN` and, when the
notebook performs Kaggle dataset operations, `KG_TOKEN`. The notebooks read
environment variables first, then attached Kaggle secrets. The code does
not read the ignored legacy `KEYS.md` or `KEYS.txt` files.

The large reader checkpoints, frozen evaluation inputs, and PLE shards are
external artifacts. To rerun a 0.8B notebook, attach the 11 pinned PLE shard
datasets listed in the [0.8B protocol](qwen35-08b/README.md), the checkpoint
and frozen evaluation datasets named in that study's report, and the account
secrets above. Check its `kernel-metadata.json` for dataset sources and GPU
settings. The [endpoint manifest](qwen35-08b/reader-scale/results/endpoints.json)
identifies the balanced 15M and max-LM 20M checkpoints; the default 20M
builder uses the [verified smaller cache](qwen35-08b/reader-scale-20m/CACHE_IO.md).
The original training notebook has its run gates off at rest. Review the
matching study report and compute cost before launching a GPU run.

Some research datasets are private. A fresh clone can check the committed
results, but checkpoint evaluation requires access to those artifacts.
Published run manifests record the original Kaggle notebook versions; the
notebooks here have different file hashes after credential cleanup. Their
frozen reader implementation cells retain the same SHA-256, and the report
checkpoint hashes remain the historical run identities. The detailed paired
outputs, alpha8 distributions, and SHA manifests are under the study
`results/` directories.

For the 35B track, configure a Modal secret named `qwen36-ple-hf` containing
`HF_TOKEN`, then follow [its README](qwen36-35b/README.md). For the historical
Lightning continuation, set `HF_TOKEN` in the Studio environment.

## Tracks

- `qwen36-35b/` — 35B target (`Qwen/Qwen3.6-35B-A3B`, hidden 2048, 40 layers)
  on one Modal A100 80GB. `modal_*.py` entrypoints, `src/qwen36_ple/`
  implementation (hashing, reader, injection, PLE store), `README.md`
  with status and measured results. Run from the repo root, e.g.
  `modal run qwen36-35b/modal_train.py`.
- `qwen35-08b/` — 0.8B target (`Qwen/Qwen3.5-0.8B`, hidden 1024, 24 layers)
  on Kaggle (recent reader runs used T4 x2). Notebook + kernel metadata + `verify_notebook.py` checker
  + `tools/` ops scripts. See its README for protocol and results.
- `qwen35-08b/ple-data/` — builder for the 11 pinned FP8 PLE shard datasets
  (`ninnix/qwen38-ple-p00` … `p10`: 33 shards, ~48.7 GiB, size+sha verified).

Shared addressing ground truth (both tracks): `vocab_size = 248320`,
`eos = 248044` (`<|endoftext|>`, the PLE-training terminator),
`seed = 1234`, `rows_per_part = 2_500_012`, 128 parts.

## Current 0.8B result

**Balanced canonical:** REAL-15M R=1 IDX2+IDX8 reader plus its separately
calibrated 749,568-token linear IDX8 arbiter. **Max-LM endpoint:** REAL-20M
R=1 plus its own linear arbiter. The 20M model improves aggregate NLL but
regresses on math against 15M, so it does not replace the balanced canonical.
The four-branch reader stopped after its first 999,936 additional tokens:
aggregate NLL improved, while code and math regressed.

### Protocol and controls

The target is `Qwen/Qwen3.5-0.8B` (hidden 1024, 24 layers). The frozen FP8
PLE contributes 16 addressed rows per token, concatenated to 2560 dimensions.
The R=1 reader has shared value projections, one key branch per injection site,
and learned residual scales at IDX2 and IDX8. Reader training used the same
FineWeb-Edu prefix; each reader milestone was saved, SHA-verified, and reopened
before evaluation. The backbone and PLE remained frozen. Each stronger reader
received a new linear dynamic IDX8 gate trained on the same balanced 500,736
then 749,568-token calibration protocol. Alpha2 stayed fixed at 1.267012596.

The frozen evaluation uses 1,024 full-val blocks, five sets of 64 domain blocks
(general, code, math, scientific, multilingual), 1,000 HellaSwag examples, and
1,000 LAMBADA examples. Reported intervals are 95% paired bootstrap intervals
from 10,000 resamples of aligned blocks or examples. NLL is lower-is-better;
domain mean weights the five domains equally. Accuracy intervals on the
1,000-example sets are much less decisive than the NLL comparisons.

At 500K reader tokens, IDX2+IDX8 improved frozen full-val NLL by 0.01362,
versus 0.00936 for IDX2 and 0.00587 for IDX8. Against the disabled, calibrated
random-memory, and permuted-memory controls, the REAL-500K full-val paired
NLL differences were respectively -0.01362 [-0.01403, -0.01323], -0.00940
[-0.00965, -0.00915], and -0.00542 [-0.00557, -0.00527]. The raw reader
continued to 5M without an early stop: full-val NLL was 2.85940 versus
2.90558 frozen. The small downstream Stage A sample remained inconclusive:
MMLU-Pro 42/238 versus 41/238 disabled; HumanEval+ 4/164 versus 5/164;
SimpleQA 0/250 for every arm. See the [0.8B study record](qwen35-08b/README.md)
for the placement, controls, Stage A, and Stage A2 details.

### Arbitration at the 5M reader

The linear IDX8 gate improved full-val and five-domain mean NLL through 1M
calibration tokens, but the 750K-to-1M leg did not consistently improve the
held-out sets: LAMBADA and four domain intervals included zero. The linear
arbiter was declared data-saturated at that point. A zero-initialized residual
MLP (RMSNorm, 1024→32, SiLU, 32→1) gave a small aggregate gain at 750K, but
LAMBADA NLL worsened versus linear750. The balanced linear750 gate was retained.

| 5M reader + gate | Full-val NLL | Five-domain mean NLL | LAMBADA NLL | HellaSwag acc |
| --- | ---: | ---: | ---: | ---: |
| Linear 319K canonical | 2.865378 | 2.395617 | 2.254326 | 0.404 |
| Linear 500K | 2.865274 | 2.395525 | 2.254049 | 0.404 |
| Linear 750K | 2.865232 | 2.395463 | 2.253318 | 0.404 |
| Linear 1M | 2.865205 | 2.395432 | 2.253382 | 0.404 |
| Residual MLP 750K | 2.865187 | 2.395388 | 2.253579 | 0.404 |

Linear 750K minus 500K full-val was -0.000041 [-0.000055, -0.000028];
linear 1M minus 750K was -0.000027 [-0.000040, -0.000014], while its
LAMBADA difference was +0.000065 [-0.000224, +0.000354]. Residual MLP 750K
minus linear 750K was -0.00004543 [-0.00005813, -0.00003256] full-val,
-0.00007522 [-0.00009565, -0.00005526] domain mean, and +0.00026150
[+0.00000095, +0.00052705] LAMBADA. The [linear report](qwen35-08b/arb-cont-kaggle/decision.md)
and [MLP report](qwen35-08b/arb-cont-kaggle/residual-results/decision.md)
include all paired contrasts and gate distributions.

### Reader scaling with a separate linear750 gate at every milestone

| Reader | Full-val NLL | Five-domain mean NLL | LAMBADA NLL | HellaSwag acc | LAMBADA acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M R=1 | 2.865232 | 2.395463 | 2.253318 | 0.404 | 0.451 |
| 7.5M R=1 | 2.860405 | 2.391695 | 2.244977 | 0.403 | 0.448 |
| 10M R=1 | 2.857079 | 2.388370 | 2.236491 | 0.402 | 0.451 |
| **15M R=1, balanced canonical** | **2.853786** | **2.384680** | **2.217277** | **0.400** | **0.456** |
| 20M R=1, max-LM | 2.851537 | 2.383358 | 2.211253 | 0.400 | 0.458 |
| 16M R=4, warm-started from 15M | 2.852958 | 2.384245 | 2.216757 | 0.399 | 0.455 |

| Reader + gate | General | Code | Math | Scientific | Multilingual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M R=1 | 3.148203 | 1.498090 | 1.413835 | 2.256121 | 3.661065 |
| 7.5M R=1 | 3.142517 | 1.496490 | 1.411149 | 2.253543 | 3.654776 |
| 10M R=1 | 3.138277 | 1.495856 | 1.406194 | 2.250134 | 3.651389 |
| **15M R=1** | **3.134094** | **1.495201** | **1.402346** | **2.244491** | **3.647268** |
| 20M R=1 | 3.129606 | 1.495631 | 1.404891 | 2.241183 | 3.645480 |
| 16M R=4 | 3.132702 | 1.495545 | 1.402984 | 2.243079 | 3.646916 |

Negative differences favor the later reader. The 15M-to-20M change improved
full-val by -0.002249 [-0.002415, -0.002085], five-domain mean by -0.001322
[-0.001648, -0.000996], and LAMBADA by -0.006023 [-0.008239, -0.003736].
The math change was **+0.002546 [+0.001627, +0.003420]**; code was +0.000430
[-0.000118, +0.001031]. The 10M-to-15M change improved full-val by -0.003293
[-0.003551, -0.003032], domain mean by -0.003690 [-0.004046, -0.003331],
and LAMBADA by -0.019215 [-0.023367, -0.015030]; four of five domain
intervals were negative, with code crossing zero. The 5M→7.5M and 7.5M→10M
steps also had negative paired intervals for full-val, domain mean, LAMBADA,
and all five domains. Full paired results are in the
[7.5M/10M](qwen35-08b/reader-scale/results/decision.md),
[15M](qwen35-08b/reader-scale-15m/results/decision.md), and
[20M](qwen35-08b/reader-scale-20m/results/decision.md) reports.

HellaSwag accuracy on the same examples moved 0.404 → 0.403 → 0.402 → 0.400
→ 0.400 from 5M through 20M. The paired five-point slope was -0.000283 per
million reader tokens, 95% CI [-0.000676, +0.000062]; it did not establish
an accuracy regression. Neither the 20M-versus-15M HellaSwag difference
(0.000 [-0.003, +0.003]) nor LAMBADA accuracy difference (+0.002
[-0.003, +0.008]) established a change.

### R=4 capacity ablation and gate behavior

The four-branch reader copied the 15M R=1 shared values and gammas, and
initialized its key branches from the R=1 keys with paired 1e-4 relative
perturbations. Its initial reader outputs differed from R=1 by at most
2.38e-7 at each site. After 999,936 extra reader tokens and a separately
calibrated linear750 gate, R=4 minus 15M R=1 was -0.000828
[-0.000889, -0.000765] full-val and -0.000435 [-0.000539, -0.000326]
domain mean. General, scientific, and multilingual improved, but **code
regressed by +0.000345 [+0.000117, +0.000574] and math by +0.000638
[+0.000391, +0.000880]**. LAMBADA NLL (-0.000519
[-0.001525, +0.000493]) and both accuracy changes were unresolved. The
[R=4 report](qwen35-08b/reader-r4/results/decision.md) records all eight
set scores, paired intervals, alpha8 distributions, and SHA manifests.

| Gate on full-val | Alpha8 mean | Std | p05 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M R=1 + linear750 | 0.121169 | 0.152115 | 0.006538 | 0.045259 | 0.486456 |
| 7.5M R=1 + linear750 | 0.122006 | 0.149851 | 0.006676 | 0.048657 | 0.483434 |
| 10M R=1 + linear750 | 0.123088 | 0.148923 | 0.006408 | 0.050890 | 0.480780 |
| 15M R=1 + linear750 | 0.124661 | 0.144834 | 0.006523 | 0.057140 | 0.471943 |
| 20M R=1 + linear750 | 0.123724 | 0.143066 | 0.006233 | 0.057927 | 0.467663 |
| 16M R=4 + linear750 | 0.124384 | 0.143500 | 0.006786 | 0.058115 | 0.469835 |

These are token-weighted absolute alpha8 statistics. Paired per-block and
per-example alpha8 deltas and CIs for mean, std, p05, p50, and p95 are in the
linked study artifacts. In a separate routing diagnostic, learned alpha8
placement beat equal-budget shuffled, hard-token-only, and easy-token-only
routing on full-val and every domain; see the [0.8B study record](qwen35-08b/README.md).

The [endpoint manifest](qwen35-08b/reader-scale/results/endpoints.json) pins
the balanced 15M and max-LM 20M reader and arbiter SHA-256 hashes. The
[canonical suffix-cache path](qwen35-08b/reader-scale-20m/CACHE_IO.md)
reproduced the 20M reader weight SHA, raw optimizer SHA, and raw full-val NLL
exactly while reducing the PLE cache from 14.668 to 5.708 GiB. The old cache
path remains a reference. The R=4 first checkpoint is a research endpoint;
no additional R=1 scaling, R=4 training, or arbiter-capacity run followed.

## Acknowledgements

- Qwen team (Alibaba Cloud) for the open models this work builds on:
  `Qwen3.8-Flash-Next` (frozen PLE source), `Qwen3.6-35B-A3B` and
  `Qwen3.5-0.8B` (frozen target backbones).
- Li et al., Cross-Model Memory Transfer via Target-Side Reader Adaptation
  (https://arxiv.org/html/2608.17050v2), for the frozen-memory transfer
  protocol, reader design, and ablations this project follows.
- Kaggle for GPU compute and dataset hosting for the 0.8B track.
- Modal for A100 80GB compute for the 35B track.
- Hugging Face for model/dataset hosting (`HuggingFaceFW/fineweb-edu` for our reader-fitting corpus).

Code in this repo is MIT licensed, see LICENSE.
