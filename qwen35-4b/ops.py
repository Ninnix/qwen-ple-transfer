import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / '.artifacts/kaggle-config'
token = os.environ.get('KAGGLE_API_TOKEN')
if not token and (CONFIG / 'access_token').exists():
    token = (CONFIG / 'access_token').read_text().strip()
if not token:
    raise SystemExit('Set KAGGLE_API_TOKEN before running Kaggle operations')
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['KAGGLE_CONFIG_DIR'] = str(CONFIG)
if sys.argv[1:2] == ['quick-save']:
    import json
    os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    job = sys.argv[2]
    assert job in ('benchmark', 'train-5m', 'train-10m', 'train-15m', 'arb-10m', 'arb-15m', 'controls')
    folder = CONFIG.parent.parent / job
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    notebook = json.loads((folder / meta['code_file']).read_text())
    for cell in notebook['cells']:
        assert not cell.get('outputs')
        if isinstance(cell['source'], list): cell['source'] = ''.join(cell['source'])
    api = KaggleApi(); api.authenticate()
    request = ApiSaveKernelRequest()
    request.slug = meta['id']; request.new_title = meta['title']
    request.text = json.dumps(notebook)
    request.language = 'python'; request.kernel_type = 'notebook'
    request.is_private = True; request.enable_gpu = True; request.enable_internet = True
    request.machine_shape = 'NvidiaTeslaT4'
    request.dataset_data_sources = meta['dataset_sources']
    request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.save_kernel(request)
    assert not result.error, result.error
    assert not result.invalid_dataset_sources, result.invalid_dataset_sources
    print('QUICK_SAVED', result.ref, result.version_number)
    raise SystemExit(0)
if sys.argv[1:2] == ['jupyter']:
    import runpy
    os.environ.update(env)
    sys.argv.pop(1)
    runpy.run_path(str(Path(__file__).with_name('jupyter.py')), run_name='__main__')
    raise SystemExit(0)
if sys.argv[1:2] == ['quota-detail']:
    import json
    os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    q = api.quota_view().gpu_quota
    print(json.dumps({k:getattr(q,k).total_seconds()/3600 for k in
                      ('time_used','time_reserved','total_time_allowed')}))
    raise SystemExit(0)
if sys.argv[1:2] in (['interactive-qualification'], ['interactive-create'], ['interactive-status'], ['interactive-cancel']):
    import json
    os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import (ApiCreateKernelSessionRequest,
                                                           ApiCancelKernelSessionRequest)
    api = KaggleApi()
    api.authenticate()
    job = sys.argv[2] if len(sys.argv) > 2 else 'benchmark'
    assert job in ('benchmark', 'train-5m', 'train-10m', 'train-15m', 'arb-10m', 'arb-15m', 'controls')
    path = CONFIG / ('qualification-session.json' if job == 'benchmark' else job + '-session.json')
    with api.build_kaggle_client() as client:
        if sys.argv[1] == 'interactive-cancel':
            state = json.loads(path.read_text())
            name = state['name']
            assert name.startswith('operations/create-kernel-session/')
            request = ApiCancelKernelSessionRequest()
            request.kernel_session_id = int(name.rsplit('/', 1)[1])
            result = client.kernels.kernels_api_client.cancel_kernel_session(request)
            assert not result.error_message, result.error_message
            state['cancel_requested'] = True
            path.write_text(json.dumps(state))
            print('CANCEL_REQUESTED', job, request.kernel_session_id)
            raise SystemExit(0)
        elif sys.argv[1] == 'interactive-status':
            state = json.loads(path.read_text())
            op = client.common.operations_client.get_operation(name=state['name'])
        else:
            assert not path.exists(), 'Inspect the existing session before creating another'
            request = ApiCreateKernelSessionRequest()
            request.slug = json.loads((CONFIG.parent.parent / job / 'kernel-metadata.json').read_text())['id']
            request.language = 'python'
            request.kernel_type = 'notebook'
            request.machine_shape = 'NvidiaTeslaT4'
            request.enable_internet = True
            op = client.kernels.kernels_api_client.create_kernel_session(request)
    previous = json.loads(path.read_text()) if path.exists() else {}
    state = {'name':op.name, 'done':op.done, 'metadata':op.metadata or previous.get('metadata'),
             'response':op.response or previous.get('response'),
             'error_code':op.error.code if op.error else None}
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as f:
        json.dump(state,f)
    print(json.dumps({'done':op.done, 'error_code':state['error_code'],
                      'metadata_keys':list(op.metadata or {}),
                      'response_keys':list(op.response or {})}))
    raise SystemExit(0)
if sys.argv[1:2] == ['pipeline']:
    import runpy
    os.environ.update(env)
    sys.argv.pop(1)
    runpy.run_path(str(Path(__file__).with_name('pipeline.py')), run_name='__main__')
    raise SystemExit(0)
if sys.argv[1:2] == ['live']:
    import json
    import signal
    os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(20)
    lines = []
    try:
        for event in api.kernels_logs_stream(sys.argv[2]):
            data = event.get('data', '').strip()
            if data:
                try:
                    data = data.encode('latin1').decode('utf8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
                lines.append(data.replace(token, '[REDACTED]'))
    except Exception as e:
        if not lines:
            print('No live log available:', type(e).__name__)
    finally:
        signal.alarm(0)
    folder = CONFIG.parent / 'live'
    folder.mkdir(exist_ok=True)
    path = folder / (sys.argv[2].split('/')[-1] + '.log')
    snapshot = path.with_suffix('.json')
    old = json.loads(snapshot.read_text()) if snapshot.exists() else []
    if lines:
        path.write_text('\n'.join(lines) + '\n')
        snapshot.write_text(json.dumps(lines))
    for line in lines[len(old):]:
        if '\x1b[' not in line:
            print(line)
    raise SystemExit(0)
if sys.argv[1:3] == ['kernels', 'push']:
    for check in ('qwen35-08b/verify_notebook.py', 'qwen35-4b/verify.py'):
        subprocess.run([sys.executable, str(ROOT / check)], cwd=ROOT, check=True)
p = subprocess.Popen([str(ROOT / '.venv/bin/kaggle'), *sys.argv[1:]],
                     cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
failed = False
for line in p.stdout:
    failed |= line.startswith('Kernel push error:')
    print(line.replace(token, '[REDACTED]'), end='', flush=True)
raise SystemExit(p.wait() or int(failed))
