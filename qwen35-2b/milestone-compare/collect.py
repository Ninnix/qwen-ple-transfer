import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1])
RESULTS = HERE / 'results'
RESULTS.mkdir(exist_ok=True)
STAGE = Path('/tmp/qwengram-2b-checkpoints')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def find(name):
    hits = list(ROOT.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]


summary = json.loads(find('summary.json').read_text())
assert summary['reader_sha256'] == '4f93cd735bd7cbe99d3d35e364b70222ee56b6c27574f12822ee58e143cd8feb'
assert summary['arb_checkpoint_sha256'] == sha(find('linear-749568.pt'))
old = json.loads((HERE.parent / 'results/arb-summary.json').read_text())
for key in ('cache_sha256', 'mapping_sha256', 'eval_streams_sha256'):
    assert summary[key] == old[key], key
checkpoint = torch.load(find('linear-749568.pt'), map_location='cpu', weights_only=False)
assert checkpoint['reader_sha256'] == summary['reader_sha256']
assert checkpoint['tokens'] == 749568 and checkpoint['optimizer_step'] == 1464
assert checkpoint['w'].shape == (2048,)
assert len(checkpoint['optimizer']['state']) == 3
manifest = {}
for name in ('eval-stock.json', 'eval-raw-10m.json', 'eval-linear750.json',
             'bootstrap.json', 'summary.json', 'stock-reproduction.json',
             'linear-500736.pt', 'linear-749568.pt',
             'qwengram-2b-10m-linear750-comparison.log'):
    src = find(name)
    if name.endswith('.pt'):
        dst = HERE.parent / '.artifacts' / ('arb10-' + name)
    else:
        dst = RESULTS / name
    shutil.copyfile(src, dst)
    manifest[name] = {'sha256': sha(dst), 'size': dst.stat().st_size,
                      'local_path': str(dst.relative_to(HERE.parent))}
(HERE / 'artifact-shas.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Verified and collected 10M arbitration and evaluation outputs')
