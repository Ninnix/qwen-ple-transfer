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
r = subprocess.run([exe, 'datasets', 'files', 'ninnix/qwen38-flashnext-ple-fp8'],
                   capture_output=True, text=True, env=env, timeout=120, cwd=REPO)
print('rc:', r.returncode)
print((r.stdout or '')[:3000])
print('ERR:', (r.stderr or '')[-500:])
