from pathlib import Path
import os, subprocess

REPO = str(Path(__file__).resolve().parents[2])
env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'

ok, missing = [], []
for i in range(11):
    slug = 'ninnix/qwen38-ple-p%02d' % i
    r = subprocess.run([exe, 'datasets', 'files', slug],
                       capture_output=True, text=True, env=env, timeout=120, cwd=REPO)
    files = sorted(l.split()[0].split('/')[-1] for l in r.stdout.splitlines()
                   if '.safetensors' in l or 'manifest.json' in l)
    want = sorted(['model-%05d-of-00131.safetensors' % (5 + 3 * i + j) for j in range(3)
                   if 5 + 3 * i + j <= 37] + ['manifest.json'])
    if files == want:
        ok.append(slug)
    else:
        missing.append((slug, files))
print('READY:', len(ok), '/11')
for s in ok:
    print('  ok', s)
for s, f in missing:
    print('  MISSING/BAD', s, f)
