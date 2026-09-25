from pathlib import Path
import os, subprocess, sys, time

REPO = str(Path(__file__).resolve().parents[2])
SUBDIR = os.path.join('qwen35-08b', 'ple-data')
CAP_MIN = 55

env = dict(os.environ)
assert env.get('KAGGLE_API_TOKEN'), 'set KAGGLE_API_TOKEN'
env.pop('KAGGLE_USERNAME', None)
env.pop('KAGGLE_KEY', None)
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
exe = 'kaggle'

def run(args, timeout=300):
    return subprocess.run([exe] + args, capture_output=True, text=True,
                          env=env, timeout=timeout, cwd=os.path.join(REPO, SUBDIR))

r = run(['kernels', 'push', '-p', '.'], timeout=600)
print('PUSH rc: %d %s' % (r.returncode, (r.stdout or '')[-400:]), flush=True)
if r.returncode != 0:
    print('PUSH STDERR:', (r.stderr or '')[-2000:], flush=True)
    sys.exit(2)

time.sleep(30)
r = run(['kernels', 'list', '--mine', '--page-size', '10'], timeout=120)
slug = None
for line in (r.stdout or '').splitlines():
    if 'ple-build' in line or 'ple-dataset-build' in line or 'flashnext-ple' in line.lower() and 'build' in line.lower():
        slug = line.split()[0]
        break
print('SLUG:', slug, flush=True)
if not slug:
    print((r.stdout or '')[-1500:], flush=True)
    sys.exit(3)

t0 = time.time()
while (time.time() - t0) < CAP_MIN * 60:
    r = run(['kernels', 'status', slug], timeout=120)
    st = ((r.stdout or '') + (r.stderr or '')).strip().replace('\n', ' ')[:250]
    print('[+%dm] %s' % (int((time.time() - t0) // 60), st), flush=True)
    low = st.lower()
    if 'complete' in low or 'error' in low or 'failed' in low or 'cancell' in low:
        break
    time.sleep(60)
print('WATCH END', flush=True)
