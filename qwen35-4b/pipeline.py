import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = HERE / '.artifacts'
RESULTS = HERE / 'results'
DEST = ART / 'checkpoints'
DATASET = 'ninnix/qwengram-4b-checkpoints'
PIN = json.loads((HERE / 'frozen.json').read_text())
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
SEEN = {5: 5000192, 10: 10000384, 15: 15000064}
JOBS = ['benchmark', 'train-5m', 'train-10m', 'train-15m', 'arb-10m', 'arb-15m', 'controls']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def kaggle(*args):
    p = subprocess.run([sys.executable, str(HERE / 'ops.py'), *args], cwd=ROOT,
                       text=True, capture_output=True, timeout=1800)
    output = p.stdout + p.stderr
    if p.returncode:
        raise RuntimeError('Kaggle %s failed: %s' % (args[0], output[-2500:]))
    return output


def find(root, name):
    hits = list(root.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]


def read(root, name):
    return json.loads(find(root, name).read_text())


def checkpoint(path, tokens, reader=True):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if reader:
        assert ck['seen'] == tokens and ck['cfg'] == ((3, 11), 1)
        assert ck['provenance']['target_revision'] == PIN['target_revision']
        assert ck['provenance']['train_stream_sha256'] == PROTOCOL['train_stream_sha256']
        assert len(ck['reader']) == len(ck['optimizer']['state']) == 8
        assert ck['corpus_cursor']['next_block'] == tokens // 512
        assert all(torch.isfinite(v).all() for v in ck['reader'].values())
    else:
        assert ck['tokens'] == tokens and ck['injection_idx'] == [3, 11]
        assert ck['target_revision'] == PIN['target_revision']
        assert len(ck['optimizer']['state']) == 3
        assert ck['w'].numel() == 2560
        assert all(torch.isfinite(ck[k]).all() for k in ('w', 'b', 'alpha2_raw'))
    return ck


def package(job, out):
    names = []
    prefix = ''
    if job == 'benchmark':
        b = read(out, 'benchmark-4b.json')
        assert b['configuration_qualified'] and b['passed_before_optimizer_step']
        assert b['execution'] == json.loads((HERE / 'execution.json').read_text())
        assert b['target_revision'] == PIN['target_revision']
        assert b['lookup_max_abs_difference'] == 0
        names = ['benchmark-4b.json', 'qualification.json', 'compact-qualification.json']
        wanted = 'benchmark-4b.json'
    elif job.startswith('train-'):
        stage = int(job.split('-')[1][:-1]); seen = SEEN[stage]
        summary = read(out, 'reader-%dm-summary.json' % stage)
        ckname = 'reader-real-%dm-%d.pt' % (stage, seen)
        weights = 'reader-%d.safetensors' % seen
        assert sha(find(out, ckname)) == summary['checkpoint_sha256']
        assert sha(find(out, weights)) == summary['reader_sha256']
        ck = checkpoint(find(out, ckname), seen)
        from safetensors.torch import load_file
        import torch
        wt = load_file(str(find(out, weights)))
        assert wt.keys() == ck['reader'].keys()
        assert all(torch.equal(wt[k], ck['reader'][k]) for k in wt)
        names = [ckname, weights, 'reader-%dm-summary.json' % stage,
                 'compact-reader-%d-%dm.json' % (stage-5, stage),
                 'reader-train-15m.json', 'evaluation-streams.json']
        wanted = ckname
    elif job.startswith('arb-'):
        stage = int(job.split('-')[1][:-1]); seen = SEEN[stage]
        summary = read(out, 'summary.json')
        assert summary['reader_sha256'] == sha(DEST / ('reader-%d.safetensors' % seen))
        assert summary['arb_checkpoint_sha256'] == sha(find(out, 'linear-749568.pt'))
        for tokens in (500736, 749568):
            ck = checkpoint(find(out, 'linear-%d.pt' % tokens), tokens, False)
            assert ck['reader_sha256'] == summary['reader_sha256']
            assert ck['reader_tokens'] == seen
        prefix = 'arb%d-' % stage
        names = ['linear-500736.pt', 'linear-749568.pt', 'eval-stock.json',
                 'eval-raw-%dm.json' % stage, 'eval-linear750.json', 'bootstrap.json',
                 'summary.json', 'compact-arbitration.json']
        wanted = prefix + 'linear-749568.pt'
    else:
        summary = read(out, 'controls-500k-summary.json')
        assert summary['budget_tokens'] == 500224
        assert set(summary['full_nll']) == {'DISABLED', 'RANDOM', 'PERMUTED', 'REAL'}
        names = ['controls-500k-summary.json', 'compact-controls-500k.json']
        for cond in ('DISABLED', 'RANDOM', 'PERMUTED', 'REAL'):
            names.append('control-eval-%s.json' % cond)
            if cond != 'DISABLED':
                ckname = 'reader-%s-500224.pt' % cond.lower()
                checkpoint(find(out, ckname), 500224)
                names += [ckname, 'reader-%s-500224.safetensors' % cond]
        prefix = 'control-'
        wanted = prefix + 'controls-500k-summary.json'
    files = json.loads((DEST / 'milestone-shas.json').read_text())
    selected = [(find(out, name), prefix + name) for name in names]
    if job != 'benchmark':
        selected.append((find(out, 'qualification.json'), job + '-qualification.json'))
    selected.append((find(out, 'runtime.json'), job + '-runtime.json'))
    selected.append((find(out, 'execution-parity.json'), job + '-execution-parity.json'))
    for path in (HERE / 'execution.json', HERE / 'execution-t4x2.json', HERE / 'checkpointing.py',
                 HERE / 'model_parallel.py', RESULTS / 'single-t4-failure.json'):
        selected.append((path, path.name))
    selected += [(p, p.name) for p in out.glob('*.log')]
    logs = list(out.glob('*.log'))
    assert len(logs) == 1
    events = json.loads(logs[0].read_text())
    submitted = list((out / 'source').glob('*.ipynb'))
    assert len(submitted) == 1
    execution = out / (job + '-execution.json')
    execution.write_text(json.dumps({'job':job, 'kernel_log_sha256':sha(logs[0]),
                                    'submitted_notebook_sha256':sha(submitted[0]),
                                    'kernel_elapsed_s':max(float(e['time']) for e in events),
                                    'execution_transport':'interactive Jupyter' if out.parent.name == 'interactive' else 'batch',
                                    'queue_time_included':False}, indent=2)+'\n')
    selected.append((execution, execution.name))
    selected.append((submitted[0], job + '-source.ipynb'))
    selected += [(p, job+'-'+p.name) for p in (out/'source').glob('*.py')]
    if (out/'remote-status.json').exists():
        selected.append((out/'remote-status.json', job+'-remote-status.json'))
    for src, name in selected:
        dst = DEST / name
        if dst.exists():
            assert sha(dst) == sha(src), name
        else:
            shutil.copyfile(src, dst)
        files[name] = {'size': dst.stat().st_size, 'sha256': sha(dst)}
        if dst.suffix == '.json':
            shutil.copyfile(dst, HERE / 'results' / name)
    (DEST / 'milestone-shas.json').write_text(json.dumps(files, indent=2) + '\n')
    return wanted


def persist(job, wanted):
    print(kaggle('datasets', 'version', '-p', str(DEST), '-m',
                 'Qwengram-4B verified ' + job)[-1500:], flush=True)
    check = ART / 'roundtrip' / job
    check.mkdir(parents=True, exist_ok=True)
    for attempt in range(40):
        listing = kaggle('datasets', 'files', DATASET, '--page-size', '200')
        if wanted in listing:
            print(kaggle('datasets', 'download', DATASET, '-f', wanted,
                         '-p', str(check), '--unzip')[-1000:], flush=True)
            assert sha(find(check, wanted)) == sha(DEST / wanted)
            print('PRIVATE_DATASET_ROUNDTRIP_VERIFIED', job, wanted, sha(DEST / wanted), flush=True)
            return
        time.sleep(30)
    raise RuntimeError('Dataset publication timed out: ' + job)


def run(job):
    meta = json.loads((HERE / job / 'kernel-metadata.json').read_text())
    slug = meta['id']
    try:
        status = kaggle('kernels', 'status', slug)
    except RuntimeError as e:
        if 'Cannot access kernel' not in str(e) and '404' not in str(e):
            raise
        while True:
            try:
                print(kaggle('kernels', 'push', '-p', str(HERE / job),
                             '--accelerator', 'NvidiaTeslaT4', '--timeout', '43200'), flush=True)
                break
            except RuntimeError as push_error:
                if 'Maximum batch GPU session count' not in str(push_error): raise
                print('Waiting for a Kaggle batch GPU session slot:', job, flush=True)
                time.sleep(60)
        status = ''
    last = None
    while True:
        status = kaggle('kernels', 'status', slug).strip()
        if status != last:
            print(status, flush=True); last = status
        if 'KernelWorkerStatus.COMPLETE' in status:
            break
        if any('KernelWorkerStatus.' + s in status for s in ('ERROR', 'CANCELLED', 'CANCEL_REQUESTED', 'CANCEL_ACKNOWLEDGED')):
            raise RuntimeError(status)
        time.sleep(30)
    out = ART / 'outputs' / job
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle('kernels', 'pull', slug, '-p', str(out / 'source'))[-1000:], flush=True)
    pattern = r'.*(benchmark-4b|execution-parity|qualification|runtime|compact-.*|reader-.*summary|evaluation-streams|reader-train-15m|eval-.*|bootstrap|summary|control-eval-.*|controls-500k-summary)\.json$|.*(reader-.*\.safetensors|reader-.*\.pt|linear-.*\.pt)$'
    for attempt in range(10):
        try:
            print(kaggle('kernels', 'output', slug, '-p', str(out), '--file-pattern', pattern)[-1500:], flush=True)
            wanted = package(job, out)
            break
        except (AssertionError, RuntimeError) as e:
            print('Output not verified:', str(e)[-1500:], flush=True)
            if attempt == 9: raise
            time.sleep(30)
    persist(job, wanted)
    return {'artifact': wanted, 'sha256': sha(DEST / wanted), 'completed_utc': time.time()}


def run_interactive(job):
    out = ART / 'interactive' / job
    log = out / (job + '.log')
    if job == 'benchmark' and '--adopt-benchmark' in sys.argv:
        print('Waiting for the executing interactive qualification', flush=True)
        while not log.exists():
            time.sleep(15)
    elif not log.exists():
        subprocess.run([sys.executable, str(HERE / 'verify.py')], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'qwen35-08b/verify_notebook.py')], cwd=ROOT, check=True)
        path = ART / 'kaggle-config' / (job + '-session.json' if job != 'benchmark' else 'qualification-session.json')
        if not path.exists():
            print(kaggle('quick-save', job), flush=True)
            print(kaggle('interactive-create', job), flush=True)
        for attempt in range(60):
            print(kaggle('interactive-status', job), flush=True)
            session = json.loads(path.read_text())
            assert not session.get('cancel_requested'), 'Session was cancelled; inspect before retrying'
            assert not session.get('error_code'), 'Kaggle session creation failed'
            if session['done']:
                break
            time.sleep(15)
        else:
            raise RuntimeError('Interactive session did not become available')
        # Stream directly to the pipeline log; a reader stage can exceed subprocess timeouts.
        result = subprocess.run([sys.executable, str(HERE / 'ops.py'), 'jupyter', 'run', job], cwd=ROOT)
        print('INTERACTIVE_EXIT', job, result.returncode, flush=True)
    print(kaggle('jupyter', 'fetch', job), flush=True)
    events = json.loads(log.read_text())
    assert events[-1]['data'] == 'NOTEBOOK_COMPLETE ' + job + '\n', 'Interactive notebook failed'
    if (out/'remote-status.json').exists():
        remote = json.loads((out/'remote-status.json').read_text())
        assert remote['status'] == 'notebook_complete' and remote['job'] == job
        assert remote['source_sha256'] == sha(HERE / job / (job+'.ipynb'))
    source = find(out / 'source', job + '.ipynb')
    assert sha(source) == sha(HERE / job / (job + '.ipynb'))
    wanted = package(job, out)
    print(kaggle('interactive-cancel', job), flush=True)
    persist(job, wanted)
    return {'artifact':wanted, 'sha256':sha(DEST / wanted), 'completed_utc':time.time(),
            'execution_transport':'Kaggle interactive Jupyter; identical frozen notebook'}


def report():
    subprocess.run([sys.executable, str(HERE / 'analyze.py')], cwd=ROOT, check=True)
    files = json.loads((DEST / 'milestone-shas.json').read_text())
    for src in (HERE / 'decision.md', HERE / 'protocol.json', HERE / 'frozen.json',
                RESULTS / 'comparison.json', RESULTS / 'cross-scale-reference.json',
                RESULTS / 'local-parity.json'):
        dst = DEST / src.name
        if dst != src: shutil.copyfile(src, dst)
        files[dst.name] = {'size':dst.stat().st_size, 'sha256':sha(dst)}
    (DEST / 'milestone-shas.json').write_text(json.dumps(files, indent=2)+'\n')
    shutil.copyfile(DEST / 'milestone-shas.json', HERE / 'results/milestone-shas.json')
    persist('final-report', 'decision.md')


if __name__ == '__main__':
    if '--detach' in sys.argv:
        with (ART / 'pipeline.log').open('a') as log:
            args = [arg for arg in sys.argv[1:] if arg != '--detach']
            p = subprocess.Popen([sys.executable, str(HERE / 'ops.py'), 'pipeline', *args], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        print('Detached 4B pipeline PID', p.pid)
        raise SystemExit(0)
    lock = (ART / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = ART / 'pipeline-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for job in JOBS:
        if job in state:
            assert sha(DEST / state[job]['artifact']) == state[job]['sha256']
            continue
        state[job] = run_interactive(job) if '--interactive' in sys.argv else run(job)
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(tmp, state_path)
    print('4B_TRAINING_AND_EVALUATION_PERSISTED', flush=True)
    report()
    print('4B_FINAL_REPORT_PERSISTED; NO FURTHER EXPERIMENTS', flush=True)
