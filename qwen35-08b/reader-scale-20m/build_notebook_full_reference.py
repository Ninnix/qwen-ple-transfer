import json
from pathlib import Path

here = Path(__file__).parent
source = here.parent / 'kaggle_qwen35_08b_ple_train.ipynb'
out = here / 'kaggle_reader_20m.ipynb'
nb = json.loads(source.read_text())

def cell(name):
    return next(c for c in nb['cells'] if c.get('id') == name)

def put(name, body):
    cell(name)['source'] = body.splitlines(keepends=True)

put('c-env', '''# Reader scale: frozen Kaggle stack and private HF credential.
import os, shutil, sys
from pathlib import Path
secret_value_0 = os.environ.get('HF_TOKEN')
if not secret_value_0:
    try:
        from kaggle_secrets import UserSecretsClient
        secret_value_0 = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        pass
assert secret_value_0 and secret_value_0.startswith('hf_'), 'set HF_TOKEN or attach Kaggle HF_TOKEN secret'
print('python', sys.version.split()[0], 'working disk', shutil.disk_usage('/kaggle/working'))
print('input mounts', sorted(str(p) for p in Path('/kaggle/input').iterdir()))
''')

ple = ''.join(cell('c-ple')['source'])
old = "hits=sorted(Path('/kaggle/input').glob('*/manifest.json'))+sorted(Path('/kaggle/input').glob('*/*/manifest.json'))"
assert ple.count(old) == 1
put('c-ple', ple.replace(old, "hits=sorted(Path('/kaggle/input').rglob('manifest.json'))"))

trainer = ''.join(cell('c-trainfn')['source'])
needle = '            v=eval_loss(inj,store,_vv,True)\n'
assert trainer.count(needle) == 1
put('c-trainfn', trainer.replace(needle, '            pre_eval_save_reader(inj,seen,th)\n' + needle))

valprep = ''.join(cell('c-valprep')['source'])
put('c-valprep', '''# Reuse and verify the exact frozen validation bytes.
import hashlib, shutil
_hits = list(Path('/kaggle/input').rglob('transfer-shas.json'))
assert len(_hits) == 1
_man = json.loads(_hits[0].read_text())
for _name in ('frozen-v1__tokens-fast.uint32le', 'frozen-v1__tokens-full.uint32le',
              'frozen-v1__validation-fast.json', 'frozen-v1__validation-full.json'):
    _src = _hits[0].parent / _name
    assert hashlib.sha256(_src.read_bytes()).hexdigest() == _man[_name]['sha256']
    _dst = VALDIR / _name.split('__', 1)[1]
    shutil.copyfile(_src, _dst)
    assert hashlib.sha256(_dst.read_bytes()).hexdigest() == _man[_name]['sha256']
''' + valprep)

compact = ''.join(cell('c-compact')['source'])
assert compact.count('500000') == 1
compact = compact.replace('500000', '20000000').replace('compact-500k', 'compact-20m')
assert compact.count('_rb=bytes(_rows.view(torch.uint8).flatten().tolist())') == 1
compact = compact.replace('_rb=bytes(_rows.view(torch.uint8).flatten().tolist())',
                          '_rb=_rows.view(torch.uint8).contiguous().numpy().tobytes()')
compact += '''\n# Release construction tensors; CompactPLE keeps only the mapped cache.\nimport gc\nfor _name in ('_train_toks', '_all', '_addrs', '_uniq', '_parts', '_local', '_rows', '_full'):\n    globals().pop(_name, None)\ngc.collect()\n'''
put('c-compact', compact)

for name in ('c-smoke', 'c-realsmoke', 'c-profile', 'c-bench', 'c-place', 'c-branch',
             'c-real1m', 'c-real5m', 'c-plot', 'c-export'):
    put(name, '# Reader scale uses the frozen R=1 IDX2+IDX8 continuation only.\n')

scale = (here / 'scale_cell.py').read_text()
idx = next(i for i, c in enumerate(nb['cells']) if c.get('id') == 'c-compact') + 1
nb['cells'].insert(idx, {'cell_type':'code', 'execution_count':None, 'id':'c-reader-scale',
                         'metadata':{}, 'outputs':[], 'source':scale.splitlines(keepends=True)})
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['execution_count'] = None
        c['outputs'] = []
out.write_text(json.dumps(nb, indent=1))
print(out)
