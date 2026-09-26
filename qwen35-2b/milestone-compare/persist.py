import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
STAGE = Path('/tmp/qwengram-2b-checkpoints')
assert (STAGE / 'reader-real-15m-15000064.pt').exists()


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


files = json.loads((STAGE / 'milestone-shas.json').read_text())
new = {}
for name, spec in json.loads((HERE / 'artifact-shas.json').read_text()).items():
    path = TRACK / spec['local_path']
    assert sha(path) == spec['sha256'] and path.stat().st_size == spec['size']
    new['arb10-' + name] = path
for name in ('protocol.json', 'comparison.json', 'decision.md', 'artifact-shas.json', 'milestone-provenance.json',
             'compare.ipynb', 'analyze.py'):
    new['milestone-comparison-' + name] = HERE / name
for name in ('README.md', 'qwengram-2b.json', 'qwengram-2b-15m.json'):
    new[name] = TRACK / name
for name in ('decision.md', 'decision-original-15m.md'):
    new[name] = TRACK / 'results' / name
for name, src in new.items():
    dst = STAGE / name
    shutil.copyfile(src, dst)
    files[name] = {'sha256': sha(dst), 'size': dst.stat().st_size}
(STAGE / 'milestone-shas.json').write_text(json.dumps(files, indent=2))
reports = json.loads((STAGE / 'report-shas.json').read_text())
for name in reports:
    reports[name] = {'sha256': sha(STAGE / name), 'size': (STAGE / name).stat().st_size}
(STAGE / 'report-shas.json').write_text(json.dumps(reports, indent=2) + '\n')
(HERE / 'persistence-shas.json').write_text(json.dumps({n: files[n] for n in new}, indent=2) + '\n')
print('Staged', len(new), 'comparison artifacts; prior checkpoint files retained')
