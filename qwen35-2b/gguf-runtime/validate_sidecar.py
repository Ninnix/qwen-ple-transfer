import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LLAMA = ROOT / 'llama.cpp'
sys.path.insert(0, str(LLAMA / 'gguf-py'))
sys.path.insert(0, str(ROOT / 'qwen36-35b/src'))
import gguf
from qwen36_ple.hashing import ngram_indices

PLE = LLAMA / 'models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
EXPECTED_SHA = '66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1'
OUT = HERE.parent / '.artifacts/gguf/sidecar'
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    before = (PLE.stat().st_size, PLE.stat().st_mtime_ns)
    assert sha(PLE) == EXPECTED_SHA
    binary = OUT / 'sidecar'
    build = LLAMA / 'build-qwengram-cpu/bin'
    cmd = ['c++', '-std=c++17', '-O2', str(HERE / 'sidecar.cpp'),
           str(LLAMA / 'src/llama-qwengram.cpp'), str(LLAMA / 'src/llama-mmap.cpp'),
           str(LLAMA / 'src/llama-impl.cpp'),
           '-I' + str(LLAMA / 'src'), '-I' + str(LLAMA / 'include'),
           '-I' + str(LLAMA / 'ggml/include'), '-L' + str(build),
           '-Wl,-rpath,' + str(build), '-lggml-base', '-o', str(binary)]
    subprocess.run(cmd, check=True)
    rng = np.random.default_rng(1234)
    tokens = rng.integers(0, 248320, size=256, dtype=np.int32)
    tokens[:8] = [248044, 248044, 1, 2, 248044, 3, 4, 248044]
    tokens[63:66] = [248044, 10, 20]
    tokens[-3:] = [248319, 248044, 248319]
    token_path = OUT / 'tokens.bin'
    tokens.tofile(token_path)
    addresses = ngram_indices(torch.tensor(tokens, dtype=torch.long).unsqueeze(0)).numpy()[0]
    reader = gguf.GGUFReader(PLE)
    weights = next(t for t in reader.tensors if t.name == 'ple.weight')
    assert weights.tensor_type == gguf.GGMLQuantizationType.Q4_1
    rows = weights.data.reshape(320001536, 100)[addresses.reshape(-1)]
    expected = gguf.dequantize(rows, weights.tensor_type).reshape(256, 2560)
    results = {}
    for chunk in (256, 17, 1):
        dst = OUT / ('rows-%d.bin' % chunk)
        subprocess.run([str(binary), str(PLE), str(token_path), str(dst), str(chunk)], check=True)
        actual = np.fromfile(dst, dtype=np.float32).reshape(2, 256, 2560)
        diff = float(np.max(np.abs(actual - expected)))
        assert diff == 0.0, (chunk, diff)
        results[str(chunk)] = {'max_abs_diff': diff, 'output_sha256': sha(dst)}
    assert before == (PLE.stat().st_size, PLE.stat().st_mtime_ns)
    result = {'ple_sha256': EXPECTED_SHA, 'tokens_sha256': sha(token_path),
              'tokens': 256, 'rows_per_pass': 4096,
              'reference_hashing_sha256': sha(ROOT / 'qwen36-35b/src/qwen36_ple/hashing.py'),
              'runtime_source_sha256': sha(LLAMA / 'src/llama-qwengram.cpp'),
              'reference': 'Unchanged PyTorch addressing + GGUF Python Q4_1 dequantization',
              'checks': 'Full prefill, split prefill, token decode, EOS boundaries, repeated position-zero reset',
              'runs': results, 'ple_unchanged': True}
    (HERE / 'sidecar-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
