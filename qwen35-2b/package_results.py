import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
kind = sys.argv[2]
assert kind in ('arbitration', 'controls')
dest = Path('/tmp/qwengram-2b-checkpoints')
assert (dest / 'reader-real-15m-15000064.pt').exists()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def find(name):
    hits = list(root.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]


files = json.loads((dest / 'milestone-shas.json').read_text())
if kind == 'arbitration':
    names = ['linear-500736.pt', 'linear-749568.pt', 'eval-stock.json',
             'eval-raw-15m.json', 'eval-linear750.json', 'bootstrap.json',
             'summary.json', 'qwengram-2b-linear750-evaluation-t4.log']
    summary = json.loads(find('summary.json').read_text())
    assert sha(find('linear-749568.pt')) == summary['arb_checkpoint_sha256']
    assert summary['reader_sha256'] == files['reader-15000064.safetensors']['sha256']
else:
    names = ['control-eval-DISABLED.json', 'control-eval-RANDOM.json',
             'control-eval-PERMUTED.json', 'control-eval-REAL.json',
             'controls-500k-summary.json', 'compact-controls-500k.json',
             'qwengram-2b-controls-500k-t4.log']
    for cond in ('random', 'permuted', 'real'):
        names += ['reader-%s-500224.pt' % cond,
                  'reader-%s-500224.safetensors' % cond.upper()]
    summary = json.loads(find('controls-500k-summary.json').read_text())
    assert summary['budget_tokens'] == 500224
    assert set(summary['full_nll']) == {'DISABLED', 'RANDOM', 'PERMUTED', 'REAL'}

for name in names:
    src = find(name)
    outname = ('arb-' + name) if kind == 'arbitration' else ('control-' + name)
    dst = dest / outname
    shutil.copyfile(src, dst)
    files[outname] = {'size': dst.stat().st_size, 'sha256': sha(dst)}
(dest / 'milestone-shas.json').write_text(json.dumps(files, indent=2))
print('packaged', kind, len(names), 'files')
