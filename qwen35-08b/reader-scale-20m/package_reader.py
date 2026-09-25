import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
source_kernel = sys.argv[3] if len(sys.argv) > 3 else 'ninnix/qwen-ple-real-r1-reader-scale-20m-p100/1'
assert source_kernel in {
    'ninnix/qwen-ple-real-r1-reader-scale-20m-p100/1',
    'ninnix/qwen-ple-real-r1-reader-scale-20m-suffix-cache-t4/1',
}

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

summary = json.loads(find('reader-20m-summary.json').read_text())
assert summary['tokens'] == 20000256
bundle = find('real-20m-r1')
assert bundle.is_dir()
files = {}
for name in ('reader.safetensors', 'reader.json', 'resume.pt', 'run.json', 'metrics.json'):
    src = bundle / name
    assert sha(src) == summary['bundle_sha256'][name]
    dst = out / ('real-20m-r1.' + name)
    shutil.copyfile(src, dst)
    files[dst.name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
assert files['real-20m-r1.reader.safetensors']['sha256'] == summary['pre_eval_reader_sha256']

for name, src in (('reader-20m-summary.json', find('reader-20m-summary.json')),
                  ('compact-20m.json', find('compact-20m.json')),
                  ('canonical-15m.json', Path('qwen35-08b/reader-scale/results/current-canonical.json'))):
    dst = out / name
    shutil.copyfile(src, dst)
    files[name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}

manifest = {'source_kernel':source_kernel,
            'files':files}
(out / 'reader-20m-shas.json').write_text(json.dumps(manifest, indent=2))
(out / 'dataset-metadata.json').write_text(json.dumps({
    'title':'Qwen PLE REAL R1 reader 20M milestone',
    'id':'ninnix/qwen-ple-reader-20m-milestone',
    'licenses':[{'name':'CC0-1.0'}], 'isPrivate':True}, indent=2))
print('packaged 20M reader milestone', {k:v['size'] for k,v in files.items()})
