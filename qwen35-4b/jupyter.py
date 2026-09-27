import json
import hashlib
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import websocket

HERE = Path(__file__).resolve().parent
job = sys.argv[2] if len(sys.argv) > 2 else 'benchmark'
assert job in ('benchmark', 'train-5m', 'train-10m', 'train-15m', 'arb-10m', 'arb-15m', 'controls')
STATE = HERE / '.artifacts/kaggle-config' / ('qualification-session.json' if job == 'benchmark' else job + '-session.json')
state = json.loads(STATE.read_text())
assert state['done'] and state['response']
base = state['response']['jupyterUrl'].rstrip('/') + '/'
session = requests.Session()


def request(method, path, **kwargs):
    try:
        response = session.request(method, base + path, timeout=60, **kwargs)
    except requests.RequestException as e:
        raise RuntimeError('Jupyter transport: ' + type(e).__name__) from None
    if not response.ok:
        raise RuntimeError('Jupyter HTTP %d: %s' % (response.status_code, path))
    return response


def connect(ident=None):
    kernels = request('GET', 'api/kernels').json()
    assert len(kernels) == 1
    ident = ident or uuid.uuid4().hex
    url = base.replace('https://', 'wss://') + 'api/kernels/' + kernels[0]['id']
    try:
        ws = websocket.create_connection(url + '/channels?session_id=' + ident, timeout=30)
    except Exception as e:
        raise RuntimeError('Jupyter WebSocket: ' + type(e).__name__) from None
    return ws, ident


def refresh_connection():
    global base
    result = subprocess.run([sys.executable, str(HERE/'ops.py'), 'interactive-status', job],
                            capture_output=True, text=True, timeout=60)
    if result.returncode: return
    state = json.loads(STATE.read_text())
    if state.get('error_code'): raise RuntimeError('Kaggle session ended: '+str(state['error_code']))
    if state.get('response'): base = state['response']['jupyterUrl'].rstrip('/')+'/'


def execute(ws, ident, code, emit, cell=None):
    msgid = uuid.uuid4().hex
    ws.send(json.dumps({'header':{'msg_id':msgid, 'username':'qwengram',
                                 'session':ident, 'date':datetime.now(timezone.utc).isoformat(),
                                 'msg_type':'execute_request', 'version':'5.3'},
                        'parent_header':{}, 'metadata':{}, 'channel':'shell',
                        'content':{'code':code, 'silent':False, 'store_history':True,
                                   'user_expressions':{}, 'allow_stdin':False,
                                   'stop_on_error':True}}))
    idle = reply = False
    error = None
    while not (idle and reply):
        try:
            raw = ws.recv()
            if not raw:
                raise websocket.WebSocketConnectionClosedException()
        except websocket.WebSocketTimeoutException:
            try: ws.ping()
            except websocket.WebSocketException: pass
            if cell and cell_finished(cell): break
            continue
        except (websocket.WebSocketException, OSError):
            ws.close()
            emit('TRANSPORT_RECONNECT; current cell will not be resubmitted\n', 'stderr')
            for attempt in range(20):
                try:
                    ws, _ = connect(ident)
                    break
                except RuntimeError:
                    if attempt == 19: raise
                    if attempt % 3 == 0: refresh_connection()
                    time.sleep(5)
            if cell and cell_finished(cell): break
            continue
        msg = json.loads(raw)
        if msg.get('parent_header', {}).get('msg_id') != msgid:
            continue
        kind = msg['header']['msg_type']
        content = msg['content']
        if kind == 'stream':
            emit(content['text'], content.get('name', 'stdout'))
        elif kind in ('execute_result', 'display_data'):
            emit(content.get('data', {}).get('text/plain', '') + '\n', 'stdout')
        elif kind == 'error':
            error = content['ename']
            emit('\n'.join(content['traceback']) + '\n', 'stderr')
        elif kind == 'status' and content['execution_state'] == 'idle':
            idle = True
        elif kind == 'execute_reply':
            reply = True
            if content['status'] == 'error': error = content['ename']
    if error:
        raise RuntimeError('Notebook cell failed: ' + error)
    return ws


def cell_finished(cell):
    try:
        status = request('GET', 'files/qwengram-cell.json').json()
    except RuntimeError:
        return False
    if status.get('job') != job or status.get('cell') != cell: return False
    if status['status'] == 'error': raise RuntimeError('Remote notebook cell failed: '+cell)
    return status['status'] in ('done', 'notebook_complete')


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'status':
        for path in ('api/status', 'api/kernels', 'api/contents'):
            data = request('GET', path).json()
            if path == 'api/contents':
                data = [{'name':v['name'], 'type':v['type']} for v in data.get('content', [])]
            print(path, json.dumps(data))
        root = request('GET', 'api/contents/qwengram-4b').json()
        for entry in root['content']:
            if entry['type'] == 'directory' and entry['name'].startswith('compact-'):
                contents = request('GET', 'api/contents/qwengram-4b/' + entry['name']).json()['content']
                print(entry['name'], json.dumps({v['name']:v['size'] for v in contents if v['type']=='file'}))
    elif action == 'test-reconnect':
        import socket
        import threading
        ws, ident = connect()
        output = []
        def emit(data, stream='stdout'):
            output.append(data); print(data, end='', flush=True)
        ws = execute(ws, ident, (HERE/'remote.py').read_text(), emit)
        timer = threading.Timer(1, lambda:ws.sock.shutdown(socket.SHUT_RDWR))
        timer.start()
        code = "_qgram_test_count = globals().get('_qgram_test_count', 0)+1\nimport time\nprint('TEST_BEGIN')\ntime.sleep(6)\nassert _qgram_test_count == 1\nprint('TEST_END')"
        ws = execute(ws, ident, '_qgram_cell(%r, %r, %r, %r)' %
                     (job, '__transport_check__', 'infrastructure-test', code), emit, '__transport_check__')
        timer.join()
        assert any('TRANSPORT_RECONNECT' in data for data in output)
        assert cell_finished('__transport_check__')
        journal = request('GET','files/qwengram-remote-events.jsonl').text
        assert journal.count('TEST_BEGIN') == journal.count('TEST_END') == 1
        assert json.loads(request('GET','files/qwengram-cell.json').text)['status'] == 'done'
        ws = execute(ws, ident, "assert _qgram_test_count == 1\nprint('EXACTLY_ONCE_CONFIRMED')", emit)
        ws.close()
        kernel = request('GET', 'api/kernels').json()[0]['id']
        request('POST', 'api/kernels/'+kernel+'/restart')
        result = {'forced_disconnect':True, 'automatic_reconnect':True,
                  'cell_executed_once':True, 'remote_log_complete':True,
                  'kernel_restarted_before_model_work':True}
        (HERE/'results/transport-recovery.json').write_text(json.dumps(result,indent=2)+'\n')
        print('TRANSPORT_RECOVERY_PASSED', json.dumps(result))
    elif action == 'probe':
        ws, ident = connect()
        try:
            execute(ws, ident, "import os, subprocess\nprint('MOUNTS',os.listdir('/kaggle/input'))\nprint(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader']).decode())",
                    lambda data, stream:print(data, end='', flush=True))
        finally:
            ws.close()
    elif action == 'fetch':
        out = HERE / '.artifacts/interactive' / job
        remote = 'qwengram-4b'
        files = request('GET', 'api/contents/' + remote).json()['content']
        for entry in files:
            name = entry['name']
            if entry['type'] != 'file' or not name.endswith(('.json', '.pt', '.safetensors')):
                continue
            assert Path(name).name == name
            dest = out / name
            tmp = dest.with_suffix(dest.suffix + '.tmp')
            h = hashlib.sha256()
            with request('GET', 'files/' + remote + '/' + name, stream=True) as response:
                with tmp.open('wb') as f:
                    for chunk in response.iter_content(8*1024*1024):
                        f.write(chunk); h.update(chunk)
                    f.flush(); os.fsync(f.fileno())
            assert tmp.stat().st_size == entry['size']
            os.replace(tmp, dest)
            print('DOWNLOADED', name, dest.stat().st_size, h.hexdigest(), flush=True)
        try:
            status = request('GET', 'files/qwengram-cell.json').json()
        except RuntimeError:
            status = None
        if status:
            assert status['job'] == job
            (out/'remote-status.json').write_text(json.dumps(status, indent=2)+'\n')
            journal = request('GET', 'files/qwengram-remote-events.jsonl').text
            (out/'remote-events.jsonl').write_text(journal)
            events = [json.loads(line) for line in journal.splitlines()]
            (out/(job+'.log')).write_text(json.dumps(events))
    elif action == 'run':
        source = HERE / job / (job + '.ipynb')
        out = HERE / '.artifacts/interactive' / job
        out.mkdir(parents=True, exist_ok=True)
        (out / 'source').mkdir(exist_ok=True)
        (out / 'source' / source.name).write_bytes(source.read_bytes())
        for name in ('jupyter.py', 'remote.py'):
            (out/'source'/name).write_bytes((HERE/name).read_bytes())
        nb = json.loads(source.read_text())
        events = []
        start = time.monotonic()
        ws, ident = connect()
        with (out / 'events.jsonl').open('a') as log:
            def emit(data, stream='stdout'):
                event = {'time':time.monotonic()-start, 'stream_name':stream, 'data':data}
                events.append(event)
                log.write(json.dumps(event)+'\n'); log.flush()
                print(data, end='', flush=True)
            try:
                ws = execute(ws, ident, (out/'source/remote.py').read_text(), emit)
                cells = [c for c in nb['cells'] if c['cell_type']=='code']
                source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                for index, cell in enumerate(cells):
                    if cell['cell_type'] != 'code': continue
                    code = '_qgram_cell(%r, %r, %r, %r, last=%r)' % (
                        job, cell['id'], source_sha, ''.join(cell['source']), index==len(cells)-1)
                    ws = execute(ws, ident, code, emit, cell['id'])
            finally:
                (out / (job + '.log')).write_text(json.dumps(events))
                ws.close()
    else:
        raise SystemExit('Unknown Jupyter action')
