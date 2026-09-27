import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ART = HERE.parent / '.artifacts/gguf'
LLAMA = ROOT / 'llama.cpp'
sys.path.insert(0, str(LLAMA / 'gguf-py'))
import gguf


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def same_tensor(a, b):
    assert a.tensor_type == b.tensor_type, a.name
    assert np.array_equal(a.shape, b.shape), a.name
    assert hashlib.sha256(memoryview(a.data)).digest() == hashlib.sha256(memoryview(b.data)).digest(), a.name


def main():
    baseline = gguf.GGUFReader(ART / 'release/QwenGram-4B-BF16.gguf')
    original = {t.name: t for t in baseline.tensors if t.name.startswith('qwengram.')}
    assert len(original) == 11
    metadata = {n: f.contents() for n, f in baseline.fields.items() if n.startswith('qwengram.')}
    result = {}
    definition = json.loads((HERE.parent / 'qwengram-4b.json').read_text())
    assert metadata['qwengram.reader_sha256'] == definition['reader_sha256']
    assert metadata['qwengram.arbiter_sha256'] == definition['arbiter_sha256']
    assert metadata['qwengram.recipe'] == definition['recipe']
    assert metadata['qwengram.reader_early_layer'] == 3
    assert metadata['qwengram.reader_late_layer'] == 11
    assert metadata['qwengram.reader_hidden_dim'] == 2560
    for precision in ('BF16', 'Q8_0', 'Q6_K', 'Q4_K_M'):
        stock_path = ART / ('stock-%s.gguf' % precision)
        qg_path = ART / 'release' / ('QwenGram-4B-%s.gguf' % precision)
        stock, qg = gguf.GGUFReader(stock_path), gguf.GGUFReader(qg_path)
        by_name = {t.name: t for t in qg.tensors}
        assert set(by_name) == {t.name for t in stock.tensors} | set(original)
        for tensor in stock.tensors:
            same_tensor(tensor, by_name[tensor.name])
        for name, tensor in original.items():
            same_tensor(tensor, by_name[name])
            assert by_name[name].tensor_type == gguf.GGMLQuantizationType.F32
        assert {n: f.contents() for n, f in qg.fields.items() if n.startswith('qwengram.')} == metadata
        tok_stock = {n: f.contents() for n, f in stock.fields.items() if n.startswith('tokenizer.')}
        tok_qg = {n: f.contents() for n, f in qg.fields.items() if n.startswith('tokenizer.')}
        assert tok_stock == tok_qg
        result[precision] = {'stock_sha256': sha(stock_path), 'qwengram_sha256': sha(qg_path),
                             'stock_bytes': stock_path.stat().st_size, 'qwengram_bytes': qg_path.stat().st_size,
                             'exact_backbone_tensors': len(stock.tensors),
                             'exact_tokenizer_fields': len(tok_stock), 'exact_fp32_extension_tensors': 11,
                             'backbone_types': dict(Counter(t.tensor_type.name for t in stock.tensors))}
    patch = subprocess.check_output(['git', '-C', str(LLAMA), 'diff', 'HEAD', '--', 'src', 'ggml'])
    result['runtime'] = {
        'base_commit': subprocess.check_output(['git', '-C', str(LLAMA), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_patch_sha256': hashlib.sha256(patch).hexdigest(),
        'models_h_sha256': sha(LLAMA / 'src/models/models.h'),
        'qwen35_cpp_sha256': sha(LLAMA / 'src/models/qwen35.cpp'),
        'libllama_sha256': sha(LLAMA / 'build-qwengram-cpu/bin/libllama.so'),
        'quantizer_sha256': sha(LLAMA / 'build-qwengram-cpu/bin/libllama-quantize-impl.so'),
    }
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    q6 = dict(result['Q6_K'], precision='Q6_K', runtime_base_commit=result['runtime']['base_commit'],
              source_patch_sha256=result['runtime']['source_patch_sha256'],
              libllama_sha256=result['runtime']['libllama_sha256'],
              quantizer_sha256=result['runtime']['quantizer_sha256'],
              source_bf16_sha256={'stock_bf16': result['BF16']['stock_sha256'],
                                 'qwengram_bf16': result['BF16']['qwengram_sha256']})
    (HERE / 'q6-verification.json').write_text(json.dumps(q6, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
