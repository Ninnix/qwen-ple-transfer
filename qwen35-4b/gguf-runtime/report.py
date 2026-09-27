import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
RELEASE = TRACK / '.artifacts/gguf/release'
PRECISIONS = {'bf16': 'BF16', 'q8': 'Q8_0', 'q6': 'Q6_K', 'q4': 'Q4_K_M'}
FORK = 'https://github.com/Ninnix/llama.cpp-qwengram'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def runtime_table(r):
    rows = ['| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | Perplexity reduction vs stock |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, precision in PRECISIONS.items():
        retention = '100%'
        if key != 'bf16':
            retention = '%.1f%% [%.1f%%, %.1f%%]' % (
                100*r['retention'][key], *[100*v for v in r['retention_95ci'][key]])
        rows.append('| %s | %.6f | %.6f | %.6f [%.6f, %.6f] | %s | %.2f%% |' % (
            precision, r['model_nll']['stock_'+key], r['model_nll']['qwengram_'+key],
            r['reader_gain_nll'][key], *r['reader_gain_95ci'][key], retention,
            r['perplexity_reduction_percent'][key]))
    return '\n'.join(rows)


def frozen_table():
    stock = json.loads((TRACK / 'results/arb15-eval-stock.json').read_text())
    gated = json.loads((TRACK / 'results/arb15-eval-linear750.json').read_text())
    rows = ['| Metric | Frozen stock | Canonical Qwengram-4B |', '| --- | ---: | ---: |',
            '| Full-validation NLL | %.6f | %.6f |' % (stock['val_nll'], gated['val_nll']),
            '| Full-validation perplexity | %.6f | %.6f |' % (math.exp(stock['val_nll']), math.exp(gated['val_nll']))]
    for domain in ('general', 'code', 'math', 'scientific', 'multilingual'):
        rows.append('| %s NLL | %.6f | %.6f |' % (domain.title(), stock['domains'][domain], gated['domains'][domain]))
    for key, title in (('domain_mean', 'Five-domain mean NLL'), ('lambada_nll', 'LAMBADA-1000 NLL')):
        rows.append('| %s | %.6f | %.6f |' % (title, stock[key], gated[key]))
    for key, title in (('lambada_acc', 'LAMBADA-1000 accuracy'), ('hs_acc', 'HellaSwag-1000 accuracy')):
        rows.append('| %s | %.1f%% | %.1f%% |' % (title, 100*stock[key], 100*gated[key]))
    return '\n'.join(rows)


def main(commit):
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    r = json.loads((HERE / 'results.json').read_text())
    llama = TRACK.parent / 'llama.cpp'
    sources = {'src/models/models.h': 'models_h_sha256', 'src/models/qwen35.cpp': 'qwen35_cpp_sha256'}
    for path, key in sources.items():
        data = subprocess.check_output(['git', '-C', str(llama), 'show', commit + ':' + path])
        assert hashlib.sha256(data).hexdigest() == r['verification']['runtime'][key], path
    assert sha(HERE / 'llama-qwengram-4b.patch') == r['verification']['runtime']['source_patch_sha256']
    (HERE / 'fork.json').write_text(json.dumps({'repo': FORK, 'commit': commit,
        'tested_runtime': r['verification']['runtime'], 'source_matches_tested_build': True}, indent=2) + '\n')
    definition = json.loads((TRACK / 'qwengram-4b.json').read_text())
    assert r['canonical'] == definition
    assert definition['recipe'] == 'REAL-15M + linear750'
    assert json.loads((TRACK / 'results/comparison.json').read_text())['canonical_reader_m'] == 15
    smoke = json.loads((HERE / 'smoke.json').read_text())
    legacy = json.loads((HERE / 'legacy-smoke.json').read_text())
    assert all(case['exact_greedy_match'] for case in legacy.values())
    hashes = {}
    for key, precision in PRECISIONS.items():
        assert r['reader_gain_95ci'][key][0] > 0
        name = 'QwenGram-4B-%s.gguf' % precision
        hashes[name] = sha(RELEASE / name)
        assert r['verification'][precision]['qwengram_sha256'] == hashes[name]
        assert r['verification'][precision]['exact_backbone_tensors'] == 441
        assert r['verification'][precision]['exact_tokenizer_fields'] == 9
        assert r['verification'][precision]['exact_fp32_extension_tensors'] == 11
        local = llama / 'models/qwengram' / name
        if local.exists():
            assert sha(local) == hashes[name]
        else:
            local.hardlink_to(RELEASE / name)
    matched = [p for p in PRECISIONS.values() if smoke[p+'-cpu']['stdout'] == smoke[p+'-vulkan']['stdout']]
    generation = ('On the tested AMD BC-250, %s with full Vulkan offload (`-ngl 99`) '
                  'matched the corresponding CPU eight-token greedy continuation for '
                  '`The capital of France is`. These short checks do not establish broad GPU parity.') % ', '.join(matched)
    unmatched = [p for p in PRECISIONS.values() if p not in matched]
    if unmatched:
        fallback = json.loads((HERE / 'q6-fallback.json').read_text())
        assert unmatched == ['Q6_K'] and fallback['exact_greedy_match']
        generation += (' Q6_K differed at full offload but matched with `-ngl 33`, '
                       'which keeps the first decoder layer on CPU. Use `-ngl 33` for Q6_K '
                       'on this tested device, or CPU (`-ngl 0`). The fallback is also a short generation check.')
    protocol = ('The matched CPU test scores 8,128 tokens from the first 64 consecutive 256-token '
                'WikiText-2 raw test chunks, scoring the last 127 tokens per chunk. All runs use '
                'eight threads and context/batch/microbatch 256, with no warmup. Reader gain is '
                '`NLL(stock) - NLL(Qwengram)`; retention divides each quantized gain by the BF16 '
                'gain. Paired 95% intervals use 10,000 resamples of 16 consecutive four-chunk '
                'blocks, seed 1234. The external PLE is Ivan Fioravanti\'s Q4_1 sidecar.')
    report = '\n\n'.join([
        '# Qwengram-4B GGUF runtime validation',
        'Canonical checkpoint: **REAL-15M + linear750**. See the [frozen decision](../decision.md) for the independent 10M/15M comparison.',
        runtime_table(r), '## Matched test', protocol,
        'BF16 has the lowest absolute Qwengram NLL in this test. Gain retention measures the added PLE benefit within each precision.',
        'All eight stock/Qwengram runs were measured with the same patched runtime. Per-chunk scores and model hashes are in [results.json](results.json); commands, timings and logs are in [logs](logs/).',
        'All 441 backbone tensors and nine tokenizer fields match exactly within each stock/Qwengram pair. All 11 reader/arbiter tensors remain bit-exact FP32 across BF16, Q8_0, Q6_K and Q4_K_M and match the verified reader and matching arbiter. See [packaging](packaging.json) and [verification](verification.json).',
        'The [PLE sidecar check](sidecar-validation.json) compares 4,096 addressed rows with independent reference addressing and dequantization. Full prefill, split prefill, token decode, EOS boundaries and repeated reset have max absolute difference zero. The PLE remains unchanged.',
        'This CPU test uses a quantized Q4_1 PLE and a WikiText-2 slice. The frozen Kaggle study uses the original FP8 PLE and different evaluation streams. These scores are separate benchmarks.',
        '## Generation checks', generation,
        'The loader rejected missing and invalid PLE sidecars and conflicting injection indices. The prior 0.8B and 2B Q8_0 CPU continuations remain exact with the updated runtime. See [smoke.json](smoke.json), [Q6 output](q6-smoke.json), [Q6 fallback](q6-fallback.json) and [legacy checks](legacy-smoke.json).',
        '## Runtime source',
        'Published fork: [%s](%s/commit/%s). The tested base commit, source patch and binary hashes are recorded in verification.json; [the source patch](llama-qwengram-4b.patch) reproduces the tested source.' % (commit, FORK, commit),
    ]) + '\n'
    (HERE / 'README.md').write_text(report)
    card = f'''---
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

Use the [Qwengram llama.cpp fork]({FORK}) at commit `{commit}`:

```sh
git clone {FORK}.git
cd llama.cpp-qwengram
git checkout {commit}
cmake -S . -B build-qwengram-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-qwengram-cpu -j --target llama-completion
export QWENGRAM_PLE=/path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
build-qwengram-cpu/bin/llama-completion -m /path/to/QwenGram-4B-Q8_0.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

For Vulkan, build with `-DGGML_VULKAN=ON`. {generation}

The fork supports the original 0.8B/2B readers at IDX2/IDX8 and the 2560-wide
4B reader at IDX3/IDX11. Hashing, row ordering and reader math are preserved.
Stock upstream llama.cpp does not execute this custom reader. MTP and
embedding-only inputs are unsupported. Vision has not been validated.

## Frozen evaluation

Canonical **REAL-15M + linear750**, using the original FP8 PLE and frozen Kaggle suite.

{frozen_table()}

All five domains improve over stock with paired NLL intervals below zero.
LAMBADA NLL improves significantly; HellaSwag and LAMBADA accuracy gains are
point estimates with intervals overlapping zero. Training used a single T4
with decoder activation checkpointing, with a 10.029 GiB measured peak.
The [study](https://github.com/Ninnix/qwen-ple-transfer/tree/main/qwen35-4b)
and [evaluation artifacts](evaluation/) preserve the protocol, per-item scores,
paired bootstrap results and checkpoint identities. The historical cross-scale
table retains the 2B canonical choice frozen when that study began.

## GGUF runtime retention

{protocol}

{runtime_table(r)}

BF16 has the lowest absolute Qwengram NLL in this test. Gain retention measures
the added PLE benefit within each precision.

All 441 backbone tensors and nine tokenizer fields match the corresponding
stock controls; all 11 extension tensors remain bit-exact FP32. See
[tensor verification](runtime/verification.json), [sidecar validation](runtime/sidecar-validation.json),
[generation checks](runtime/smoke.json) and the [matched report](runtime/README.md).
The Q4_1 runtime test and the FP8 frozen evaluation above are separate benchmarks.

The target model revision is `{definition['backbone_revision']}`.
Full provenance is in [qwengram-4b.json](qwengram-4b.json), `evaluation/`, and `runtime/`.
Study-relative paths in the checkpoint definition resolve in the linked study repository.
This is an experimental text-generation release.
'''
    (HERE / 'HF-README.md').write_text(card)
    (RELEASE / 'README.md').write_text(card)
    evaluation = RELEASE / 'evaluation'
    runtime = RELEASE / 'runtime'
    evaluation.mkdir(exist_ok=True)
    runtime.mkdir(exist_ok=True)
    for name in ('decision.md', 'frozen.json', 'protocol.json', 'execution.json'):
        shutil.copyfile(TRACK / name, evaluation / name)
    excluded = {'monitor.json', 'monitor-status.json', 'operations.json', 'training-progress.json'}
    for path in (TRACK / 'results').glob('*.json'):
        if path.name not in excluded:
            dst = evaluation / 'results' / path.name
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, dst)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.json', '.py', '.cpp', '.patch', '.md') and path.name not in ('HF-README.md', 'publication.json'):
            shutil.copyfile(path, runtime / path.name)
    p = runtime / 'README.md'
    p.write_text(p.read_text().replace('(../decision.md)', '(../evaluation/decision.md)'))
    shutil.copytree(HERE / 'logs', runtime / 'logs', dirs_exist_ok=True)
    files = {}
    for path in sorted(RELEASE.rglob('*')):
        if path.is_file() and path.name != 'SHA256.json':
            name = str(path.relative_to(RELEASE))
            files[name] = {'sha256': hashes[name] if name in hashes else sha(path),
                           'size': path.stat().st_size}
    (RELEASE / 'SHA256.json').write_text(json.dumps({'canonical': definition, 'files': files}, indent=2) + '\n')
    print('Prepared', len(files), 'hashed release files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-commit', required=True)
    main(parser.parse_args().runtime_commit)
