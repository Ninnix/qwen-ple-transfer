import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def find(name):
    hits = list(root.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]

summary = json.loads(find('reader-15m-summary.json').read_text())
assert summary['tokens'] == 15000064
bundle = find('real-15m-r1')
assert bundle.is_dir()
files = {}
for name in ('reader.safetensors', 'reader.json', 'resume.pt', 'run.json', 'metrics.json'):
    src = bundle / name
    assert sha(src) == summary['bundle_sha256'][name]
    dst = out / ('real-15m-r1.' + name)
    shutil.copyfile(src, dst)
    files[dst.name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
assert files['real-15m-r1.reader.safetensors']['sha256'] == summary['pre_eval_reader_sha256']

for name, src in (('reader-15m-summary.json', find('reader-15m-summary.json')),
                  ('canonical-10m.json', Path('qwen35-08b/reader-scale/results/current-canonical.json'))):
    dst = out / name
    shutil.copyfile(src, dst)
    files[name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}

manifest = {'source_kernel':'ninnix/qwen-ple-real-r1-reader-scale-15m-p100/1',
            'files':files}
(out / 'reader-15m-shas.json').write_text(json.dumps(manifest, indent=2))
(out / 'dataset-metadata.json').write_text(json.dumps({
    'title':'Qwen PLE REAL R1 reader 15M milestone',
    'id':'ninnix/qwen-ple-reader-15m-milestone',
    'licenses':[{'name':'CC0-1.0'}], 'isPrivate':True}, indent=2))
print('packaged 15M reader milestone', {k:v['size'] for k,v in files.items()})
