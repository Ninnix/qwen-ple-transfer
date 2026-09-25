from pathlib import Path
import os, subprocess

REPO = str(Path(__file__).resolve().parents[2])
SLUG = 'ninnix/qwen3-5-0-8b-ple-target-side-reader-p100'
OUT = os.path.join(REPO, 'tmp', 'v16.log')
os.makedirs(OUT if not OUT.endswith('.log') else os.path.dirname(OUT), exist_ok=True)

env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'
r = subprocess.run([exe, 'kernels', 'logs', SLUG], capture_output=True,
                   env=env, timeout=240, cwd=REPO)
print('rc:', r.returncode, '| stdout bytes:', len(r.stdout or b''))
if r.returncode == 0 and r.stdout:
    open(OUT, 'wb').write(r.stdout)
    print('saved', OUT)
