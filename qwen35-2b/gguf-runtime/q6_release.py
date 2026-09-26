import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'tmp/qwengram-q6'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def row(r, key, name):
    retention = '100%'
    if key != 'bf16':
        retention = '%.1f%% [%.1f%%, %.1f%%]' % (100*r['retention'][key],
                    *[100*v for v in r['retention_95ci'][key]])
    return '| %s | %.6f | %.6f | %.6f [%.6f, %.6f] | %s | %.2f%% |' % (
        name, r['model_nll']['stock_'+key], r['model_nll']['qwengram_'+key],
        r['reader_gain_nll'][key], *r['reader_gain_95ci'][key], retention,
        r['perplexity_reduction_percent'][key])


def table(r):
    lines = ['| Precision | Stock NLL | Qwengram NLL | Reader gain [95% CI] | Gain retention [95% CI] | Perplexity reduction vs stock |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [row(r, k, n) for k, n in [('bf16', 'BF16'), ('q8', 'Q8_0'), ('q6', 'Q6_K'), ('q4', 'Q4_K_M')]]
    return '\n'.join(lines)


def add_row(text, r):
    text = re.sub(r'^\| Q6_K \|.*\n', '', text, flags=re.M)
    return re.sub(r'(^\| Q4_K_M \|)', lambda m: row(r, 'q6', 'Q6_K') + '\n' + m.group(), text, count=1, flags=re.M)


def prepare(label):
    here = ROOT / ('qwen35-' + label) / 'gguf-runtime'
    before = ART / 'hf-input' / label
    out = ART / 'hf-output' / label
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads((here / 'results.json').read_text())
    spec = json.loads((here / 'q6-verification.json').read_text())
    smoke = json.loads((here / 'q6-smoke.json').read_text())
    assert r['reader_gain_95ci']['q6'][0] > 0
    assert smoke['exact_greedy_match'], 'Review GPU mismatch before publishing'
    assert spec['exact_backbone_tensors'] == 335 and spec['exact_fp32_extension_tensors'] == 11
    size = '0.8B' if label == '08b' else '2B'
    name = 'QwenGram-%s-Q6_K.gguf' % size
    source = ART / label / name
    assert sha(source) == spec['qwengram_sha256']
    target = out / name
    if not target.exists():
        target.hardlink_to(source)
    assert sha(target) == spec['qwengram_sha256']
    validation = ('Q6_K was quantized and tested with fork commit `%s`. All 335 backbone tensors '
                  'and nine tokenizer fields match the stock Q6_K control; all 11 reader and arbiter '
                  'tensors remain bit-exact FP32. CPU and full Vulkan offload (`-ngl 99`) on the '
                  'tested AMD BC-250 produced identical eight-token greedy continuations for '
                  '`The capital of France is`. This short generation check does not establish '
                  'general GPU parity. See [Q6 verification](q6-verification.json) and '
                  '[generation outputs](q6-smoke.json).') % spec['runtime_commit']
    text = (here / 'README.md').read_text()
    text = add_row(text, r)
    text = re.sub(r'\n## Q6_K extension\n.*', '', text, flags=re.S)
    text += '\n## Q6_K extension\n\n' + validation + '\n'
    (here / 'README.md').write_text(text)
    runtime = out / 'runtime'
    runtime.mkdir(exist_ok=True)
    for file in ('README.md', 'results.json', 'analyze.py', 'q6-verification.json', 'q6-smoke.json'):
        shutil.copyfile(here / file, runtime / file)
    if label == '2b':
        p = runtime / 'README.md'
        p.write_text(p.read_text().replace('(../milestone-compare/decision.md)', '(../evaluation/milestone-decision.md)'))
    (runtime / 'logs').mkdir(exist_ok=True)
    for p in (here / 'logs').iterdir():
        if label == '08b' or 'q6' in p.name.lower():
            shutil.copyfile(p, runtime / 'logs' / p.name)
    card = (before / 'README.md').read_text()
    if label == '08b':
        card = card.replace('| `QwenGram-0.8B-Q4_K_M.gguf`',
                            '| `QwenGram-0.8B-Q6_K.gguf` | Q6_K quantized text backbone; all 11 reader and gate tensors remain bit-exact FP32. |\n| `QwenGram-0.8B-Q4_K_M.gguf`')
        card = add_row(card, r)
        card = card.replace('The quantized GGUF files were generated directly', 'The original Q8_0 and Q4_K_M GGUF files were generated directly')
        card = card.replace('Stock Q8_0 and Q4_K_M were made', 'The original stock Q8_0 and Q4_K_M controls were made')
        card = card.replace('[reproducible CPU test](https://github.com/Ninnix/qwen-ple-transfer/tree/main/qwen35-08b/gguf-runtime)', '[reproducible CPU test](runtime/README.md)')
        card = card.replace('\n## Provenance', '\n' + validation.replace('(q6-', '(runtime/q6-') + '\n\n## Provenance')
    else:
        definition = json.loads((here.parent / 'qwengram-2b.json').read_text())
        assert definition['reader_tokens'] == 10000384 and definition['recipe'] == 'REAL-10M + linear750'
        stock = json.loads((here.parent / 'milestone-compare/results/eval-stock.json').read_text())
        gated = json.loads((here.parent / 'milestone-compare/results/eval-linear750.json').read_text())
        frozen = ['## Frozen evaluation', '', 'Canonical **REAL-10M + linear750**, using the original FP8 PLE and frozen Kaggle evaluation suite.', '',
                  '| Metric | Frozen stock | Canonical Qwengram-2B |', '| --- | ---: | ---: |',
                  '| Full-validation NLL | %.6f | %.6f |' % (stock['val_nll'], gated['val_nll']),
                  '| Full-validation perplexity | %.6f | %.6f |' % (math.exp(stock['val_nll']), math.exp(gated['val_nll']))]
        for domain in ('general', 'code', 'math', 'scientific', 'multilingual'):
            frozen.append('| %s NLL | %.6f | %.6f |' % (domain.title(), stock['domains'][domain], gated['domains'][domain]))
        for key, title in [('domain_mean', 'Five-domain mean NLL'), ('lambada_nll', 'LAMBADA-1000 NLL')]:
            frozen.append('| %s | %.6f | %.6f |' % (title, stock[key], gated[key]))
        for key, title in [('lambada_acc', 'LAMBADA-1000 accuracy'), ('hs_acc', 'HellaSwag-1000 accuracy')]:
            frozen.append('| %s | %.1f%% | %.1f%% |' % (title, 100*stock[key], 100*gated[key]))
        frozen += ['', 'These frozen study metrics are not Q6_K measurements. Quantized GGUF retention is evaluated separately below.', '']
        card = card.replace('`QwenGram-2B-Q8_0.gguf` and', '`QwenGram-2B-Q8_0.gguf`, `QwenGram-2B-Q6_K.gguf` and')
        card = card.replace('base commit\n`ab878ab4ddb578698b2b39ba1b2f0b36f07385d7`, with the included [2B patch](runtime/llama-qwengram-2b.patch):',
                            'commit\n`%s`, which includes 2B support:' % spec['runtime_commit'])
        card = card.replace('git checkout ab878ab4ddb578698b2b39ba1b2f0b36f07385d7\ngit apply /path/to/runtime/llama-qwengram-2b.patch', 'git checkout ' + spec['runtime_commit'])
        card = card.replace('The patch supports', 'The fork supports')
        section = '\n'.join(frozen) + '\n## GGUF runtime retention\n\n'
        section += ('The matched CPU test scores 8,128 tokens from the first 64 consecutive 256-token WikiText-2 raw test chunks. '
                    'All runs use eight threads and context/batch/microbatch 256. Reader gain is `NLL(stock) - NLL(Qwengram)`; '
                    'retention divides each quantized gain by the BF16 gain. Paired 95% intervals use 10,000 resamples '
                    'of 16 consecutive four-chunk blocks, seed 1234. The external PLE is Ivan Fioravanti\'s Q4_1 sidecar.\n\n')
        section += table(r) + '\n\n' + validation.replace('(q6-', '(runtime/q6-') + '\n\n'
        card = card.replace('## Runtime validation\n\n', section)
    (out / 'README.md').write_text(card)
    (here / 'HF-README.md').write_text(card)
    manifest = json.loads((before / 'SHA256.json').read_text())
    size_field = 'bytes' if label == '08b' else 'size'
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name != 'SHA256.json':
            manifest['files'][str(p.relative_to(out))] = {size_field: p.stat().st_size, 'sha256': sha(p)}
    (out / 'SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in out.rglob('*'):
        if p.is_file() and p.suffix != '.gguf':
            data = p.read_bytes()
            assert not re.search(rb'(?:hf_|github_pat_|ghp_|KGAT_)[A-Za-z0-9_-]{16,}', data), str(p)
    print(label, 'release prepared', name, flush=True)


def publish(label):
    here = ROOT / ('qwen35-' + label) / 'gguf-runtime'
    before = json.loads((ART / 'hf-input' / label / 'revision.json').read_text())
    out = ART / 'hf-output' / label
    files = sorted(p for p in out.rglob('*') if p.is_file())
    api = HfApi()
    assert api.whoami()['name'].lower() == 'ninnix96'
    manifest = json.loads((out / 'SHA256.json').read_text())
    for p in files:
        if p.name != 'SHA256.json':
            assert sha(p) == manifest['files'][str(p.relative_to(out))]['sha256']
    commit = api.create_commit(repo_id=before['repo'], parent_commit=before['commit'],
                              commit_message='Add verified Q6_K GGUF and matched runtime retention',
                              operations=[CommitOperationAdd(path_in_repo=str(p.relative_to(out)), path_or_fileobj=p) for p in files])
    info = api.model_info(before['repo'], revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    verified = {}
    for p in files:
        name = str(p.relative_to(out))
        entry = remote[name]
        digest = sha(p)
        assert entry.size == p.stat().st_size, name
        if entry.lfs:
            assert entry.lfs.sha256 == digest, name
            method = 'HF content-addressed LFS SHA256 and byte size'
        else:
            downloaded = Path(hf_hub_download(before['repo'], name, revision=commit.oid, token=False,
                                             local_dir=ART / 'remote-verify' / label))
            assert sha(downloaded) == digest, name
            method = 'Downloaded content SHA256 and byte size'
        verified[name] = {'sha256': digest, 'bytes': p.stat().st_size, 'verification': method}
    result = {'repo': before['repo'], 'commit': commit.oid, 'verified_files': verified}
    (here / 'q6-publication.json').write_text(json.dumps(result, indent=2) + '\n')
    print(before['repo'], commit.oid, len(verified), 'files verified', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare', 'publish'))
    p.add_argument('model', choices=('08b', '2b'))
    a = p.parse_args()
    try:
        (prepare if a.stage == 'prepare' else publish)(a.model)
    except Exception as error:
        token = get_token()
        message = str(error)
        if token:
            message = message.replace(token, '[REDACTED]')
        raise SystemExit(message) from None
