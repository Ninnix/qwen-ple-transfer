from pathlib import Path
import os, subprocess, sys, time

REPO = str(Path(__file__).resolve().parents[2])
SLUG = 'ninnix/qwen3-5-0-8b-ple-target-side-reader-p100'
OUT = os.path.join(REPO, 'tmp', 'kaggle_run_out')
os.makedirs(OUT if not OUT.endswith('.log') else os.path.dirname(OUT), exist_ok=True)
CAP_MIN = 60

env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'

def run(args, timeout=300):
    return subprocess.run([exe] + args, capture_output=True, text=True,
                          env=env, timeout=timeout, cwd=REPO)

r = run(['kernels', 'push', '-p', 'qwen35-08b'], timeout=600)
print('PUSH rc: %d %s' % (r.returncode, (r.stdout or '')[-500:]), flush=True)
if r.returncode != 0:
    print('PUSH STDERR:', (r.stderr or '')[-2000:], flush=True)
    sys.exit(2)

t0 = time.time()
final = None
while (time.time() - t0) < CAP_MIN * 60:
    try:
        r = run(['kernels', 'status', SLUG], timeout=120)
    except Exception as e:
        print('[+%dm] local net error, retrying: %s' % (int((time.time() - t0) // 60), str(e)[:120]), flush=True)
        time.sleep(60)
        continue
    st = ((r.stdout or '') + (r.stderr or '')).strip().replace('\n', ' ')[:300]
    el = int((time.time() - t0) // 60)
    print('[+%dm] %s' % (el, st), flush=True)
    low = st.lower()
    if 'complete' in low and 'status' in low:
        final = ('COMPLETE', st)
        break
    if 'kernelworkerstatus.error' in low or 'status "failed"' in low or 'cancelled' in low:
        final = ('NOT-OK', st)
        break
    time.sleep(60)

if final is None:
    print('WATCH CAP REACHED, still running (or queued).', flush=True)
else:
    os.makedirs(OUT, exist_ok=True)
    r = run(['kernels', 'output', SLUG, '-p', OUT, '--force'], timeout=600)
    print('OUTPUT FETCH rc:', r.returncode, flush=True)
    n = 0
    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            p = os.path.join(root, f)
            print('OUT FILE:', os.path.relpath(p, OUT), os.path.getsize(p), flush=True)
            n += 1
    print('FINAL:', final[0], '|', final[1], flush=True)
