import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from run import execute, sha
from verify import same_tensor
import gguf
import smoke

ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / 'llama.cpp'
BIN = LLAMA / 'build-qwengram-cpu/bin'
PLE = LLAMA / 'models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
ART = ROOT / 'tmp/qwengram-q6'
TEXT = ROOT / 'qwen35-2b/.artifacts/gguf/wiki.test.raw'


def paths(label):
    size = '0.8B' if label == '08b' else '2B'
    track = ROOT / ('qwen35-' + label)
    stock = ROOT / ('tmp/qwengram-08b/backbone-bf16.gguf' if label == '08b'
                    else 'qwen35-2b/.artifacts/gguf/stock-BF16.gguf')
    folder = ART / label
    folder.mkdir(parents=True, exist_ok=True)
    return track / 'gguf-runtime', stock, LLAMA / ('models/qwengram/QwenGram-%s-BF16.gguf' % size), folder


def verify(label, here, stock, reader, folder):
    old = json.loads((here / 'results.json').read_text())
    sources = old['gguf_sha256'] if label == '08b' else {
        'stock_bf16': old['verification']['BF16']['stock_sha256'],
        'qwengram_bf16': old['verification']['BF16']['qwengram_sha256']}
    assert sha(stock) == sources['stock_bf16']
    assert sha(reader) == sources['qwengram_bf16']
    baseline = gguf.GGUFReader(reader)
    extensions = {t.name: t for t in baseline.tensors if t.name.startswith('qwengram.')}
    assert len(extensions) == 11
    a = gguf.GGUFReader(folder / 'stock-Q6_K.gguf')
    b = gguf.GGUFReader(folder / reader.name.replace('BF16', 'Q6_K'))
    tensors = {t.name: t for t in b.tensors}
    assert set(tensors) == {t.name for t in a.tensors} | set(extensions)
    for tensor in a.tensors:
        same_tensor(tensor, tensors[tensor.name])
    for name, tensor in extensions.items():
        same_tensor(tensor, tensors[name])
        assert tensors[name].tensor_type == gguf.GGMLQuantizationType.F32
    def fields(model, prefix):
        return {k: v.contents() for k, v in model.fields.items() if k.startswith(prefix)}
    assert fields(a, 'tokenizer.') == fields(b, 'tokenizer.') == fields(baseline, 'tokenizer.')
    assert fields(b, 'qwengram.') == fields(baseline, 'qwengram.')
    assert subprocess.check_output(['git', '-C', str(LLAMA), 'diff', 'HEAD', '--', 'src', 'ggml']) == b''
    result = {
        'precision': 'Q6_K', 'source_bf16_sha256': sources,
        'stock_sha256': sha(folder / 'stock-Q6_K.gguf'),
        'qwengram_sha256': sha(folder / reader.name.replace('BF16', 'Q6_K')),
        'qwengram_bytes': (folder / reader.name.replace('BF16', 'Q6_K')).stat().st_size,
        'exact_backbone_tensors': len(a.tensors), 'exact_fp32_extension_tensors': len(extensions),
        'exact_tokenizer_fields': len(fields(a, 'tokenizer.')),
        'backbone_types': dict(Counter(t.tensor_type.name for t in a.tensors)),
        'runtime_commit': subprocess.check_output(['git', '-C', str(LLAMA), 'rev-parse', 'HEAD'], text=True).strip(),
        'libllama_sha256': sha(BIN / 'libllama.so'),
        'quantizer_sha256': sha(BIN / 'libllama-quantize-impl.so'),
        'dataset_sha256': sha(TEXT),
    }
    assert result['dataset_sha256'] == old['dataset_sha256']
    (here / 'q6-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(label, 'Q6 tensors, metadata, tokenizer and source hashes verified', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=('08b', '2b'))
    p.add_argument('stage', choices=('prepare', 'evaluate', 'smoke'))
    args = p.parse_args()
    here, stock, reader, folder = paths(args.model)
    env = os.environ.copy()
    env['QWENGRAM_PLE'] = str(PLE)
    model = folder / reader.name.replace('BF16', 'Q6_K')
    if args.stage == 'prepare':
        for arm, source, target in [('stock', stock, folder / 'stock-Q6_K.gguf'),
                                    ('qwengram', reader, model)]:
            log = here / 'logs' / ('quant-%s-q6.log' % arm)
            assert not target.exists(), 'Refusing to overwrite ' + str(target)
            execute([BIN / 'llama-quantize', '--tensor-type', '^qwengram[.]=f32',
                     source, target, 'Q6_K', '8'], log, env)
        verify(args.model, here, stock, reader, folder)
    elif args.stage == 'evaluate':
        spec = json.loads((here / 'q6-verification.json').read_text())
        assert sha(TEXT) == spec['dataset_sha256']
        for arm, path in [('stock', folder / 'stock-Q6_K.gguf'), ('qwengram', model)]:
            assert sha(path) == spec[arm + '_sha256']
            log = here / 'logs' / ('%s-q6.log' % arm)
            assert not log.exists(), 'Refusing to overwrite ' + str(log)
            run_env = env.copy()
            if arm == 'stock':
                run_env.pop('QWENGRAM_PLE')
            execute([BIN / 'llama-perplexity', '-m', path, '-f', TEXT,
                     '-c', '256', '-b', '256', '-ub', '256', '--chunks', '64',
                     '--ppl-output-type', '1', '--no-warmup', '-t', '8', '-ngl', '0'], log, run_env)
    else:
        smoke.HERE = here
        devices = subprocess.check_output([str(LLAMA / 'build-qwengram-vulkan/bin/llama-completion'),
                                           '--list-devices'], text=True)
        assert 'Vulkan0:' in devices, 'Vulkan device unavailable'
        results = {'vulkan_devices': devices}
        for backend, ngl in [('cpu', 0), ('vulkan', 99)]:
            r = smoke.run('Q6_K-' + backend, model, backend, ngl)
            assert r['returncode'] == 0, 'Generation failed; see smoke log'
            results[backend] = {'stdout': r['stdout'], 'command': r['command']}
            print(args.model, backend, repr(r['stdout']), flush=True)
        results['exact_greedy_match'] = results['cpu']['stdout'] == results['vulkan']['stdout']
        (here / 'q6-smoke.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
