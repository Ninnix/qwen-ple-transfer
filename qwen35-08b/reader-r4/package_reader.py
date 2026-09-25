import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
source_kernel = 'ninnix/qwen-ple-real-r4-reader-warm-start-early-1m-t4/3'

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

summary = json.loads(find('reader-r4-summary.json').read_text())
assert summary['tokens'] == 16000000 and summary['additional_tokens'] == 999936
bundle = find('real-r4-early')
assert bundle.is_dir()
files = {}
for name in ('reader.safetensors', 'reader.json', 'resume.pt', 'run.json', 'metrics.json'):
    src = bundle / name
    assert sha(src) == summary['bundle_sha256'][name]
    dst = out / ('real-r4-early.' + name)
    shutil.copyfile(src, dst)
    files[dst.name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
assert files['real-r4-early.reader.safetensors']['sha256'] == summary['pre_eval_reader_sha256']
for name, src in (('reader-r4-summary.json', find('reader-r4-summary.json')),
                  ('compact-r4.json', find('compact-r4.json')),
                  ('canonical-15m.json', Path('qwen35-08b/reader-scale/results/current-canonical.json'))):
    dst = out / name
    shutil.copyfile(src, dst)
    files[name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
manifest = {'source_kernel':source_kernel, 'files':files}
(out / 'reader-r4-shas.json').write_text(json.dumps(manifest, indent=2))
(out / 'dataset-metadata.json').write_text(json.dumps({
    'title':'Qwen PLE REAL R4 warm-start early reader milestone',
    'id':'ninnix/qwen-ple-reader-r4-early-1m-milestone',
    'licenses':[{'name':'CC0-1.0'}], 'isPrivate':True}, indent=2))
print('packaged R4 reader milestone', {k:v['size'] for k,v in files.items()})
