import os
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if '--detach' in sys.argv:
    with open('/tmp/qwengram-2b-finish.log', 'a') as log:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    print('detached pipeline', p.pid, flush=True)
    raise SystemExit(0)
lock = open('/tmp/qwengram-2b-finish.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
KAGGLE = ROOT / '.venv/bin/kaggle'
key = os.environ.get('KAGGLE_API_TOKEN')
if not key:
    raise SystemExit('Set KAGGLE_API_TOKEN before running the finish pipeline')
env = os.environ.copy()
env['KAGGLE_CONFIG_DIR'] = '/tmp/kaggle-config'
DATASET = 'ninnix/qwengram-2b-checkpoints'
JOBS = {
    'arbitration': {
        'slug': 'ninnix/qwengram-2b-linear750-evaluation-t4',
        'path': HERE / 'arb-eval',
        'pattern': r'.*(linear-[0-9]+\.pt|eval-.*\.json|bootstrap\.json|summary\.json)$',
    },
    'controls': {
        'slug': 'ninnix/qwengram-2b-controls-500k-t4',
        'path': HERE / 'controls',
        'pattern': r'.*(control-eval-.*\.json|controls-500k-summary\.json|reader-(random|permuted|real)-500224\.pt|reader-(RANDOM|PERMUTED|REAL)-500224\.safetensors|compact-controls-500k\.json)$',
    },
}


def kaggle(*args):
    p = subprocess.run([str(KAGGLE), *args], cwd=ROOT, env=env,
                       text=True, capture_output=True)
    output = (p.stdout + p.stderr).replace(key, '[REDACTED]')
    if p.returncode:
        raise RuntimeError('kaggle %s failed: %s' % (args[0], output[-2500:]))
    return output


def wait_reader():
    wanted = 'reader-real-15m-15000064.pt'
    while True:
        files = kaggle('datasets', 'files', DATASET, '--page-size', '200')
        if wanted in files:
            print('READER_15M_DATASET_READY', flush=True)
            return
        time.sleep(30)


def wait_kernel(slug):
    last = None
    while True:
        status = kaggle('kernels', 'status', slug).strip()
        if status != last:
            print(status, flush=True)
            last = status
        if 'KernelWorkerStatus.COMPLETE' in status:
            return
        if any(s in status for s in ('KernelWorkerStatus.ERROR', 'KernelWorkerStatus.CANCELLED')):
            raise RuntimeError(status)
        time.sleep(30)


def start_or_resume(cfg):
    try:
        status = kaggle('kernels', 'status', cfg['slug']).strip()
    except RuntimeError as e:
        if 'Cannot access kernel' not in str(e):
            raise
        print(kaggle('kernels', 'push', '-p', str(cfg['path']))[-1200:], flush=True)
        return
    print('RESUME', status, flush=True)
    if not any('KernelWorkerStatus.' + s in status for s in ('RUNNING', 'QUEUED', 'COMPLETE')):
        raise RuntimeError(status)


def retrieve(kind, cfg):
    out = Path('/tmp/qwengram-2b-%s-output' % kind)
    out.mkdir(exist_ok=True)
    expected = 'summary.json' if kind == 'arbitration' else 'controls-500k-summary.json'
    for attempt in range(10):
        try:
            print(kaggle('kernels', 'output', cfg['slug'], '-p', str(out),
                         '--file-pattern', cfg['pattern'])[-1000:], flush=True)
            if list(out.rglob(expected)):
                return out
        except Exception as e:
            print(str(e)[-1000:], flush=True)
        time.sleep(30)
    raise RuntimeError('%s output unavailable' % kind)


wait_reader()
for kind, cfg in JOBS.items():
    wanted = 'arb-linear-749568.pt' if kind == 'arbitration' else 'control-controls-500k-summary.json'
    if wanted in kaggle('datasets', 'files', DATASET, '--page-size', '200'):
        print('ALREADY_PERSISTED', wanted, flush=True)
        continue
    start_or_resume(cfg)
    wait_kernel(cfg['slug'])
    out = retrieve(kind, cfg)
    p = subprocess.run([sys.executable, str(HERE / 'package_results.py'),
                        str(out), kind], cwd=ROOT, text=True, capture_output=True)
    print(p.stdout + p.stderr, flush=True)
    if p.returncode:
        raise RuntimeError('%s packaging failed' % kind)
    print(kaggle('datasets', 'version', '-p', '/tmp/qwengram-2b-checkpoints',
                 '-m', 'Qwengram-2B %s results and checkpoints' % kind)[-1200:], flush=True)
    for attempt in range(40):
        listing = kaggle('datasets', 'files', DATASET, '--page-size', '200')
        if wanted in listing:
            print('DATASET_PERSISTED', wanted, flush=True)
            break
        time.sleep(30)
    else:
        raise RuntimeError('private checkpoint dataset did not publish %s' % wanted)
print('FULL_PIPELINE_COMPLETE', flush=True)
