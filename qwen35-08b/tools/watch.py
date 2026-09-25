from pathlib import Path
import os, subprocess, sys, time

REPO = str(Path(__file__).resolve().parents[2])
SLUG = 'ninnix/qwen3-5-0-8b-ple-target-side-reader-p100'
TRIES = int(sys.argv[1]) if len(sys.argv) > 1 else 8

env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'

for i in range(TRIES):
    try:
        r = subprocess.run([exe, 'kernels', 'status', SLUG], capture_output=True,
                           text=True, env=env, timeout=90, cwd=REPO)
        st = ((r.stdout or '') + (r.stderr or '')).strip().replace('\n', ' ')[:250]
        print('[%s] %s' % (time.strftime('%H:%M:%S'), st), flush=True)
        low = st.lower()
        if 'complete' in low and 'status' in low:
            break
        if 'kernelworkerstatus.error' in low or 'status "failed"' in low or 'cancell' in low:
            break
    except Exception as e:
        print('[%s] local net error, retrying: %s' % (time.strftime('%H:%M:%S'), str(e)[:120]), flush=True)
    time.sleep(150)
