import hashlib
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
RELEASE = TRACK / '.artifacts/gguf/release'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    r = json.loads((HERE / 'results.json').read_text())
    definition = r['canonical']
    assert definition == json.loads((TRACK / 'qwengram-2b.json').read_text())
    assert definition['reader_tokens'] == 10000384
    assert r['reader_gain_95ci']['bf16'][0] > 0
    assert r['reader_gain_95ci']['q8'][0] > 0
    smoke = json.loads((HERE / 'smoke.json').read_text())
    for precision in ('BF16', 'Q8_0', 'Q4_K_M'):
        assert smoke[precision+'-cpu']['stdout'] == smoke[precision+'-vulkan']['stdout']
    lines = ['# Qwengram-2B GGUF runtime validation', '',
             'Canonical checkpoint: **%s**. The [milestone comparison](../milestone-compare/decision.md) '
             'retains 15M separately as the max-LM/research endpoint.' % definition['recipe'], '',
             '| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | PPL reduction |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, name in (('bf16', 'BF16'), ('q8', 'Q8_0'), ('q4', 'Q4_K_M')):
        gain = r['reader_gain_nll'][key]
        lo, hi = r['reader_gain_95ci'][key]
        retention = '100%'
        if key != 'bf16':
            retention = '%.1f%% [%.1f%%, %.1f%%]' % (100*r['retention'][key], *[100*v for v in r['retention_95ci'][key]])
        lines.append('| %s | %.6f | %.6f | %.6f [%.6f, %.6f] | %s | %.3f%% |' %
                     (name, r['model_nll']['stock_'+key], r['model_nll']['qwengram_'+key],
                      gain, lo, hi, retention, r['perplexity_reduction_percent'][key]))
    lines += ['', '## Matched test', '',
              'The original 0.8B runtime protocol is reused: first 64 consecutive 256-token '
              'WikiText-2 raw test chunks, scoring the last 127 tokens per chunk (8,128 total). '
              'All six runs use the same CPU build, eight threads, context/batch/microbatch 256, '
              'and no warmup. Confidence intervals use 10,000 paired resamples of 16 consecutive '
              'four-chunk blocks, seed 1234. Per-chunk values and exact file hashes are in '
              '[results.json](results.json); commands and logs are in [logs](logs/).', '',
              'All 335 backbone tensors and tokenizer fields match exactly within each '
              'stock/Qwengram pair. All 11 reader/arbiter tensors remain bit-exact FP32 '
              'through BF16, Q8_0 and Q4_K_M. See [verification](verification.json).', '',
              'The Q4_1 PLE sidecar is host mapped and has SHA256 '
              '`66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1`. '
              'The [sidecar validation](sidecar-validation.json) checks 4,096 addressed rows '
              'against unchanged PyTorch addressing and independent Python Q4_1 dequantization: '
              'full prefill, split prefill and single-token decoding all have max absolute '
              'difference 0, including EOS boundaries and sequence reset.', '',
              'This small CPU test uses a Q4_1 PLE. The Kaggle study uses the original FP8 PLE, '
              'FP16 backbone execution and different frozen evaluation streams. These scores '
              'are separate results, not a numerical reproduction of the Kaggle benchmark.', '',
              '## Generation checks', '',
              'On the tested AMD BC-250 Vulkan device, all three 2B precisions with full offload '
              '(`-ngl 99`) matched their CPU eight-token greedy continuation for '
              '`The capital of France is`, beginning with ` Paris.`. The original 0.8B Q8_0 '
              'CPU smoke also passed. The loader rejected missing and invalid PLE sidecars. '
              'See [smoke.json](smoke.json). This short check is not a general GPU parity claim.', '']
    (HERE / 'README.md').write_text('\n'.join(lines))
    evaluation = RELEASE / 'evaluation'
    runtime = RELEASE / 'runtime'
    research = RELEASE / 'research-15m'
    for folder in (evaluation, runtime, research):
        folder.mkdir(exist_ok=True)
    shutil.copytree(TRACK / 'milestone-compare/results', evaluation / '10m', dirs_exist_ok=True)
    for name in ('decision.md', 'comparison.json', 'protocol.json', 'milestone-provenance.json',
                 'artifact-shas.json', 'persistence-verification.json'):
        shutil.copyfile(TRACK / 'milestone-compare' / name, evaluation / ('milestone-' + name))
    for name in ('arb-eval-stock.json', 'arb-eval-raw-15m.json', 'arb-eval-linear750.json',
                 'arb-bootstrap.json', 'decision-original-15m.md', 'evaluation-streams.json'):
        shutil.copyfile(TRACK / 'results' / name, evaluation / name)
    for name in ('README.md', 'results.json', 'verification.json', 'sidecar-validation.json',
                 'source.json', 'packaging.json', 'smoke.json', 'llama-qwengram-2b.patch'):
        shutil.copyfile(HERE / name, runtime / name)
    p = evaluation / 'milestone-decision.md'
    p.write_text(p.read_text().replace('(protocol.json)', '(milestone-protocol.json)')
                 .replace('(comparison.json)', '(milestone-comparison.json)').replace('(results/)', '(10m/)'))
    p = runtime / 'README.md'
    p.write_text(p.read_text().replace('(../milestone-compare/decision.md)', '(../evaluation/milestone-decision.md)'))
    shutil.copytree(HERE / 'logs', runtime / 'logs', dirs_exist_ok=True)
    src = TRACK / '.artifacts/qwengram-2b-remote-verify'
    shutil.copyfile(src / 'reader-15000064.safetensors', research / 'reader.safetensors')
    shutil.copyfile(src / 'arb-linear-749568.pt', research / 'arbiter.pt')
    shutil.copyfile(TRACK / 'qwengram-2b-15m.json', research / 'qwengram-2b.json')
    commit = r['verification']['runtime']['base_commit']
    card = '''---
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

`QwenGram-2B-BF16.gguf`, `QwenGram-2B-Q8_0.gguf` and `QwenGram-2B-Q4_K_M.gguf`
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

Use the [Qwengram llama.cpp fork](https://github.com/Ninnix/llama.cpp), base commit
`__COMMIT__`, with the included [2B patch](runtime/llama-qwengram-2b.patch):

```sh
git clone https://github.com/Ninnix/llama.cpp.git
cd llama.cpp
git checkout __COMMIT__
git apply /path/to/runtime/llama-qwengram-2b.patch
cmake -S . -B build-qwengram-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-qwengram-cpu -j --target llama-completion
export QWENGRAM_PLE=/path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf
build-qwengram-cpu/bin/llama-completion -m /path/to/QwenGram-2B-Q8_0.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

For Vulkan, build with `-DGGML_VULKAN=ON` and use `-ngl 99`. On the tested AMD
BC-250, BF16, Q8_0 and Q4_K_M matched their CPU eight-token greedy smoke output
with full offload. This is a short generation check, not broad GPU parity.

The patch supports both the original 0.8B reader and the 2048-wide 2B reader.
It changes reader dimensions and the corresponding inverse-square-root scale;
injection placement, hashing, PLE lookup and arbitration semantics stay the same.
Stock upstream llama.cpp does not execute this custom reader. MTP and
embedding-only inputs are unsupported for this Qwengram runtime.

## Runtime validation

See the [matched stock-versus-Qwengram runtime report](runtime/README.md).
It uses a fixed WikiText-2 slice and a quantized Q4_1 sidecar; the Kaggle scores
above use the original FP8 PLE and the frozen study suite. Do not treat them as
the same benchmark. The runtime sidecar reproduces reference Q4_1 lookups
exactly for the tested prefill, decode, EOS and reset cases.

The target model revision is `15852e8c16360a2fea060d615a32b45270f8a8fc`.
Full provenance is in `qwengram-2b.json`, the evaluation artifacts, and `runtime/`.
This is an experimental text-generation release; vision has not been validated.
'''.replace('__COMMIT__', commit)
    (RELEASE / 'README.md').write_text(card)
    files = {str(p.relative_to(RELEASE)): {'sha256': sha(p), 'size': p.stat().st_size}
             for p in sorted(RELEASE.rglob('*')) if p.is_file() and p.name != 'SHA256.json'}
    (RELEASE / 'SHA256.json').write_text(json.dumps({'canonical': definition, 'files': files}, indent=2) + '\n')
    print('Release card and', len(files), 'verified artifact records prepared')


if __name__ == '__main__':
    main()
