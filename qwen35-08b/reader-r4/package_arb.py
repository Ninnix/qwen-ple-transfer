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

names = ['summary.json', 'contrast.json', 'checkpoints.json', 'alpha8-paired.json',
         'eval-reader-r4-arb750.json', 'alpha8-reader-r4-arb750.pt',
         'arb-reader-r4-arb500-500736.pt', 'arb-reader-r4-arb750-749568.pt']
files = {}
for name in names:
    src = root / name
    dst = out / name
    shutil.copyfile(src, dst)
    files[name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
checks = json.loads((out / 'checkpoints.json').read_text())
assert files['arb-reader-r4-arb500-500736.pt']['sha256'] == checks['arbitration_shas']['reader-r4-arb500']
assert files['arb-reader-r4-arb750-749568.pt']['sha256'] == checks['arbitration_shas']['reader-r4-arb750']
manifest = {'source_kernel':'ninnix/qwen-ple-real-r4-early-arbitration-t4/1',
            'reader_source':checks['reader_source'],'files':files}
(out / 'arb-r4-shas.json').write_text(json.dumps(manifest, indent=2))
(out / 'dataset-metadata.json').write_text(json.dumps({
    'title':'Qwen PLE REAL R4 early arbitration results',
    'id':'ninnix/qwen-ple-real-r4-early-arbitration-results',
    'licenses':[{'name':'CC0-1.0'}], 'isPrivate':True}, indent=2))
print('packaged R4 arbitration', len(files))
