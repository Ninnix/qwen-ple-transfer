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

names = ['summary.json', 'contrasts.json', 'checkpoints.json', 'alpha8-paired.json']
names += [p.name for pattern in ('eval-reader*.json', 'alpha8-reader*.pt', 'arb-reader*.pt')
          for p in sorted(root.glob(pattern))]
assert len(names) == 18, names
files = {}
for name in names:
    src = root / name
    dst = out / name
    shutil.copyfile(src, dst)
    files[name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}

checks = json.loads((out / 'checkpoints.json').read_text())
for label, digest in checks['arbitration_shas'].items():
    tokens = 500736 if label.endswith('500') else 749568
    assert files['arb-%s-%d.pt' % (label, tokens)]['sha256'] == digest

manifest = {'source_kernel':'ninnix/qwen-ple-reader-scale-arbitration-p100/2',
            'reader_source':checks['reader_source'], 'files':files}
(out / 'arb-result-shas.json').write_text(json.dumps(manifest, indent=2))
(out / 'dataset-metadata.json').write_text(json.dumps({
    'title':'Qwen PLE reader scale arbitration results',
    'id':'ninnix/qwen-ple-reader-scale-arbitration-results',
    'licenses':[{'name':'CC0-1.0'}], 'isPrivate':True}, indent=2))
print('packaged', len(files), 'arbitration result files')
