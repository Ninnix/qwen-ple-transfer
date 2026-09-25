from pathlib import Path
import os, subprocess, time

REPO = str(Path(__file__).resolve().parents[2])
SLUG = 'ninnix/qwen3-5-0-8b-ple-target-side-reader-p100'
OUT = os.path.join(REPO, 'tmp', 'kaggle_run_out')
os.makedirs(OUT if not OUT.endswith('.log') else os.path.dirname(OUT), exist_ok=True)

env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'

for attempt in range(6):
    r = subprocess.run([exe, 'kernels', 'output', SLUG, '-p', OUT, '--force'],
                       capture_output=True, text=True, env=env, timeout=300,
                       cwd=REPO)
    log = os.path.join(OUT, 'qwen3-5-0-8b-ple-target-side-reader-p100.log')
    if r.returncode == 0 and os.path.exists(log) and os.path.getsize(log) > 0:
        print('LOG OK bytes:', os.path.getsize(log))
        break
    print('attempt %d rc=%d err=%s' % (attempt, r.returncode, (r.stderr or '')[-300:]))
    time.sleep(30)
