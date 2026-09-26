import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
ROOT = TRACK.parent
sys.path.insert(0, str(ROOT / 'llama.cpp/gguf-py'))
import gguf

ART = TRACK / '.artifacts/gguf'
OUT = ART / 'release'
SOURCE = TRACK / '.artifacts/qwengram-2b-remote-verify'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    definition = json.loads((TRACK / 'qwengram-2b.json').read_text())
    assert 'pending' not in definition['status'].lower()
    selected = json.loads((TRACK / 'milestone-compare/comparison.json').read_text())['canonical_balanced_milestone_m']
    assert definition['recipe'] == 'REAL-%dM + linear750' % selected
    reader_path = SOURCE / definition['reader_file']
    gate_path = TRACK / '.artifacts' / definition['arbiter_file']
    if not gate_path.exists():
        gate_path = SOURCE / definition['arbiter_file']
    assert sha(reader_path) == definition['reader_sha256']
    assert sha(gate_path) == definition['arbiter_sha256']
    reader = load_file(reader_path)
    gate = torch.load(gate_path, map_location='cpu', weights_only=False)
    assert gate['tokens'] == 749568
    assert gate['reader_sha256'] == definition['reader_sha256']
    assert gate['injection_idx'] == [2, 8]
    assert gate['target_revision'] == definition['backbone_revision']
    assert gate['tokenizer_sha256'] == definition['tokenizer_sha256']
    keys = {'readers.%d.%s' % (layer, name) for layer in (2, 8)
            for name in ('keys.0.weight', 'value.weight', 'beta', 'gamma')}
    assert set(reader) == keys
    for layer in (2, 8):
        for name in ('keys.0.weight', 'value.weight'):
            assert reader['readers.%d.%s' % (layer, name)].shape == (2048, 2560)
    assert gate['w'].shape == (2048,)
    alpha = 0.75 + torch.sigmoid(gate['alpha2_raw'])
    assert abs(alpha.item() - definition['early_alpha']) < 1e-7
    tensors = {'qwengram.' + k: v.contiguous() for k, v in reader.items()}
    tensors.update({'qwengram.arbiter.w': gate['w'].contiguous(),
                    'qwengram.arbiter.b': gate['b'].contiguous(),
                    'qwengram.arbiter.alpha2': alpha})
    assert len(tensors) == 11
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in tensors.values())
    base = gguf.GGUFReader(ART / 'stock-BF16.gguf')
    assert base.get_field('general.architecture').contents() == 'qwen35'
    assert base.get_field('qwen35.embedding_length').contents() == 2048
    assert base.get_field('qwen35.block_count').contents() - base.get_field('qwen35.nextn_predict_layers').contents() == 24
    path = OUT / 'QwenGram-2B-BF16.gguf'
    temp = path.with_suffix('.gguf.tmp')
    writer = gguf.GGUFWriter(temp, 'qwen35')
    for name, field in base.fields.items():
        if name.startswith('GGUF.') or name == 'general.architecture':
            continue
        value = field.contents()
        if name == 'general.name':
            value = 'QwenGram-2B REAL-%dM linear750' % selected
        writer.add_key_value(name, value, field.types[0],
                             field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    strings = {'version': '1', 'base_revision': definition['backbone_revision'],
               'ple_revision': definition['ple_revision'],
               'reader_sha256': definition['reader_sha256'],
               'arbiter_sha256': definition['arbiter_sha256'],
               'tokenizer_sha256': definition['tokenizer_sha256'],
               'recipe': definition['recipe'], 'reader_layers': 'IDX2,IDX8 (zero-based)',
               'alpha8': '0.5*sigmoid(dot(w,RMSNorm(h8))+b)',
               'ple_source': definition['ple_model'],
               'ple_addressing': 'splitmix64 n-gram hash, head-preserving prime tables',
               'reader_math': 'h + alpha*gamma*sigmoid(dot(RMSNorm(h),RMSNorm(key(m)))/sqrt(2048)+beta)*value(m)'}
    for name, value in strings.items():
        writer.add_string('qwengram.' + name, value)
    integers = {'reader_branches': 1, 'reader_memory_dim': 2560, 'reader_hidden_dim': 2048,
                'ple_seed': 1234, 'ple_ngram': 3, 'ple_heads_per_ngram': 8,
                'ple_vocab_base': 20000000, 'ple_vocab_size': 248320, 'ple_eos': 248044,
                'ple_row_dim': 160, 'ple_slots': 16,
                'reader_tokens': definition['reader_tokens'], 'calibration_tokens': 749568}
    for name, value in integers.items():
        writer.add_uint32('qwengram.' + name, value)
    writer.add_float32('qwengram.alpha2', alpha.item())
    writer.add_float32('qwengram.ple_weight_scale', 0.00019931793212890625)
    writer.add_bool('qwengram.external_ple_required', True)
    for tensor in base.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    for name, value in tensors.items():
        writer.add_tensor(name, value.numpy().reshape(-1) if value.ndim == 0 else value.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with temp.open('rb') as f:
        os.fsync(f.fileno())
    check = gguf.GGUFReader(temp)
    by_name = {t.name: t for t in check.tensors}
    assert len(by_name) == len(base.tensors) + 11
    for tensor in base.tensors:
        other = by_name[tensor.name]
        assert other.tensor_type == tensor.tensor_type
        assert np.array_equal(other.shape, tensor.shape)
        assert np.array_equal(other.data, tensor.data), tensor.name
    for name, value in tensors.items():
        other = by_name[name]
        assert other.tensor_type == gguf.GGMLQuantizationType.F32
        assert np.array_equal(other.data, value.numpy().reshape(other.data.shape)), name
    os.replace(temp, path)
    for name in ('LICENSE', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json'):
        shutil.copyfile(ART / 'base' / name, OUT / name)
    shutil.copyfile(TRACK / 'qwengram-2b.json', OUT / 'qwengram-2b.json')
    shutil.copyfile(reader_path, OUT / 'reader.safetensors')
    shutil.copyfile(gate_path, OUT / 'arbiter.pt')
    result = {'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
              'reader_sha256': definition['reader_sha256'], 'arbiter_sha256': definition['arbiter_sha256'],
              'stock_backbone_sha256': sha(ART / 'stock-BF16.gguf'),
              'exact_backbone_tensors': len(base.tensors), 'exact_fp32_extension_tensors': 11}
    (HERE / 'packaging.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
