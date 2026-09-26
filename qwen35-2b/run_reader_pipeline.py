import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
KAGGLE = ROOT / '.venv/bin/kaggle'
key = os.environ.get('KAGGLE_API_TOKEN')
if not key:
    raise SystemExit('Set KAGGLE_API_TOKEN before running the reader pipeline')
env = os.environ.copy()
env['KAGGLE_CONFIG_DIR'] = '/tmp/kaggle-config'
DATASET = 'ninnix/qwengram-2b-checkpoints'
SLUGS = {5: 'ninnix/qwengram-2b-reader-0-5m-t4',
         10: 'ninnix/qwengram-2b-reader-5-10m-t4',
         15: 'ninnix/qwengram-2b-reader-10-15m-t4'}
SEEN = {5: 5000192, 10: 10000384, 15: 15000064}
PATTERN = r'.*(reader-[0-9]+\.safetensors|reader-real-[0-9]+m-[0-9]+\.pt|reader-[0-9]+m-summary\.json|reader-train-15m\.(uint32le|json)|evaluation-streams\.json|compact-reader-.*\.json)$'


def kaggle(*args):
    p = subprocess.run([str(KAGGLE), *args], cwd=ROOT, env=env,
                       text=True, capture_output=True)
    output = (p.stdout + p.stderr).replace(key, '[REDACTED]')
    if p.returncode:
        raise RuntimeError('kaggle %s failed: %s' % (args[0], output[-2500:]))
    return output


def wait_kernel(stage):
    slug = SLUGS[stage]
    last = None
    while True:
        output = kaggle('kernels', 'status', slug).strip()
        if output != last:
            print(output, flush=True)
            last = output
        if 'KernelWorkerStatus.COMPLETE' in output:
            return
        if any(s in output for s in ('KernelWorkerStatus.ERROR', 'KernelWorkerStatus.CANCELLED')):
            raise RuntimeError('reader %dM kernel failed: %s' % (stage, output))
        time.sleep(30)


def retrieve(stage):
    out = Path('/tmp/qwengram-2b-reader-%dm-output' % stage)
    out.mkdir(exist_ok=True)
    for attempt in range(10):
        try:
            print(kaggle('kernels', 'output', SLUGS[stage], '-p', str(out),
                         '--file-pattern', PATTERN)[-1000:], flush=True)
            if list(out.rglob('reader-%dm-summary.json' % stage)):
                break
        except Exception as e:
            print(str(e)[-1000:], flush=True)
        time.sleep(30)
    else:
        raise RuntimeError('reader %dM output unavailable' % stage)
    p = subprocess.run([sys.executable, str(HERE / 'package_checkpoint.py'),
                        str(out), str(stage)], cwd=ROOT, text=True,
                       capture_output=True)
    print(p.stdout + p.stderr, flush=True)
    if p.returncode:
        raise RuntimeError('reader %dM packaging failed' % stage)
    return out


def persist(stage):
    print(kaggle('datasets', 'version', '-p', '/tmp/qwengram-2b-checkpoints',
                 '-m', 'Qwengram-2B reader %dM verified recovery checkpoint' % stage)[-1200:], flush=True)
    wanted = 'reader-real-%dm-%d.pt' % (stage, SEEN[stage])
    for attempt in range(40):
        listing = kaggle('datasets', 'files', DATASET, '--page-size', '200')
        if wanted in listing:
            print('DATASET_PERSISTED', stage, wanted, flush=True)
            return
        time.sleep(30)
    raise RuntimeError('private checkpoint dataset did not publish %s' % wanted)


def push(stage):
    print(kaggle('kernels', 'push', '-p', str(HERE / ('train-%dm' % stage)))[-1200:], flush=True)


start = int(sys.argv[1]) if len(sys.argv) > 1 else 5
assert start in (5, 10, 15)
for stage in (5, 10, 15):
    if stage < start:
        continue
    wait_kernel(stage)
    retrieve(stage)
    persist(stage)
    if stage == 5:
        push(10)
    if stage == 10:
        push(15)
print('READER_15M_PERSISTED', flush=True)
