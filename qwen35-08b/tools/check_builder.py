import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / 'ple-data'
nb = json.loads((D / 'build_ple_dataset.ipynb').read_text())
code = {c['id']: ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'}
assert set(code) == {'b-setup', 'b-payload', 'b-create', 'b-loop', 'b-final'}, set(code)
assert code['b-payload'].count('model-00') >= 33 + 100, 'shards+manifest refs'
assert '"0": "model-00005-of-00131.safetensors"' in code['b-payload']
assert '"127": "model-00037-of-00131.safetensors"' in code['b-payload']
assert 'SHARDS' in code['b-payload'] and 'len(SHARDS)' in code['b-loop']
assert 'for i in range(11)' in code['b-create'], 'plan loop'
assert "slug='ninnix/qwen38-ple-'+tag" in code['b-loop'] and "'datasets','create'" in code['b-loop'], 'per-batch dataset create'
assert 'exists, skip' in code['b-loop'], 'idempotent reruns'
assert 'sha256 mismatch' in code['b-loop'], 'sha verification'
assert 'torchaudio' not in ''.join(code.values()), 'no stray deps'
km = json.loads((D / 'kernel-metadata.json').read_text())
assert km['enable_gpu'] == 'false' and km['enable_internet'] == 'true'
assert km['code_file'] == 'build_ple_dataset.ipynb'
print('BUILDER OK:', km['id'], '| cells:', sorted(code))
