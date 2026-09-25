from pathlib import Path
import os, subprocess

REPO = str(Path(__file__).resolve().parents[2])
env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
exe = 'kaggle'

r = subprocess.run([exe, '--version'], capture_output=True, text=True, env=env, timeout=60)
print('cli:', (r.stdout or r.stderr).strip()[:200])
r = subprocess.run([exe, 'kernels', 'list', '--mine', '--page-size', '1'],
                   capture_output=True, text=True, env=env, timeout=120)
print('list-mine rc:', r.returncode)
out = (r.stdout or '')[:1500]
err = (r.stderr or '')[-1500:]
if out:
    print(out)
if r.returncode != 0:
    print('STDERR:', err)
