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

summary = find('reader-scale-summary.json')
reader = json.loads(summary.read_text())
files = {}
for label in ('7p5m', '10m'):
    bundle = find('real-' + label + '-r1')
    assert bundle.is_dir()
    for base, suffix in (('reader.safetensors', 'reader.safetensors'),
                         ('reader.json', 'reader.json'),
                         ('resume.pt', 'resume.pt'), ('run.json', 'run.json'),
                         ('metrics.json', 'metrics.json')):
        src = bundle / base
        assert sha(src) == reader[label]['bundle_sha256'][base]
        name = 'real-' + label + '-r1.' + suffix
        dst = out / name
        shutil.copyfile(src, dst)
        files[name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}

dst = out / 'reader-scale-summary.json'
shutil.copyfile(summary, dst)
files[dst.name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
old = json.loads(Path('qwen35-08b/arb-cont-kaggle/results/summary.json').read_text())['749568']
dst = out / 'linear-750-eval.json'
dst.write_text(json.dumps(old))
files[dst.name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
manifest = {'source_kernel':'ninnix/qwen-ple-reader-scale-7p5m-10m-p100/3', 'files':files}
(out / 'reader-stage-shas.json').write_text(json.dumps(manifest, indent=2))
shutil.copyfile('qwen35-08b/reader-scale/reader-stage-metadata.json', out / 'dataset-metadata.json')
print('packaged reader milestones', {k:v['size'] for k,v in files.items()})
