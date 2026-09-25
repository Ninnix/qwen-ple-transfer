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

def run(args, timeout=120):
    r = subprocess.run([exe] + args, capture_output=True, text=True,
                       env=env, timeout=timeout, cwd=REPO)
    return r.returncode, ((r.stdout or '') + (r.stderr or ''))[-2500:]

for args in (['datasets', 'list', '--mine', '--page-size', '20'],
             ['datasets', 'list', '--search', 'flash-next ple', '--page-size', '10'],
             ['datasets', 'list', '--search', 'qwen ple ngram', '--page-size', '10']):
    print('### kaggle', ' '.join(args))
    rc, out = run(args)
    print('rc:', rc)
    print(out)
    print()
