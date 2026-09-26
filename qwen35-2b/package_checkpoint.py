import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = int(sys.argv[2])
assert stage in (5, 10, 15)
dest = Path('/tmp/qwengram-2b-checkpoints')
assert dest.exists() and (dest / 'dataset-metadata.json').exists()
seen = {5: 5000192, 10: 10000384, 15: 15000064}[stage]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def find(name):
    hits = list(root.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]


summary_path = find('reader-%dm-summary.json' % stage)
summary = json.loads(summary_path.read_text())
checkpoint_name = 'reader-real-%dm-%d.pt' % (stage, seen)
weights_name = 'reader-%d.safetensors' % seen
assert sha(find(checkpoint_name)) == summary['checkpoint_sha256']
assert sha(find(weights_name)) == summary['reader_sha256']
train_manifest = find('reader-train-15m.json') if stage == 5 else dest / 'reader-train-15m.json'
assert summary['train_sha256'] == json.loads(train_manifest.read_text())['sha256']

names = [summary_path.name, checkpoint_name, weights_name, 'evaluation-streams.json']
names.append({5: 'qwengram-2b-reader-0-5m-t4.log',
              10: 'qwengram-2b-reader-5-10m-t4.log',
              15: 'qwengram-2b-reader-10-15m-t4.log'}[stage])
if stage == 5:
    names += ['reader-train-15m.uint32le', 'reader-train-15m.json']
compact_name = 'compact-reader-%s.json' % ({5: '0-5m', 10: '5-10m', 15: '10-15m'}[stage])
names.append(compact_name)
files = json.loads((dest / 'milestone-shas.json').read_text()) if (dest / 'milestone-shas.json').exists() else {}
for name in names:
    src = find(name)
    if name == 'reader-train-15m.uint32le':
        assert sha(src) == summary['train_sha256']
    dst = dest / name
    if dst.exists():
        assert sha(dst) == sha(src), name
    else:
        shutil.copyfile(src, dst)
    files[name] = {'size': dst.stat().st_size, 'sha256': sha(dst)}
assert files[checkpoint_name]['sha256'] == summary['checkpoint_sha256']
assert files[weights_name]['sha256'] == summary['reader_sha256']
(dest / 'milestone-shas.json').write_text(json.dumps(files, indent=2))
print('packaged', stage, 'M', len(names), 'files', 'checkpoint', files[checkpoint_name]['sha256'])
