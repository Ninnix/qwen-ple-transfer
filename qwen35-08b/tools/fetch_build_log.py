from pathlib import Path
import os, subprocess

REPO = str(Path(__file__).resolve().parents[2])
SLUG = 'ninnix/qwen3-8-flash-next-ple-dataset-build'
OUT = os.path.join(REPO, 'tmp', 'kaggle_build_out')
os.makedirs(OUT if not OUT.endswith('.log') else os.path.dirname(OUT), exist_ok=True)

env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'
os.makedirs(OUT, exist_ok=True)
r = subprocess.run([exe, 'kernels', 'output', SLUG, '-p', OUT, '--force'],
                   capture_output=True, text=True, env=env, timeout=600, cwd=REPO)
print('rc:', r.returncode)
import pathlib
for p in sorted(pathlib.Path(OUT).rglob('*')):
    if p.is_file():
        print(p.name, p.stat().st_size)
