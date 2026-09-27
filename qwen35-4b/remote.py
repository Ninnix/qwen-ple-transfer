import contextlib as _qgram_context
import json as _qgram_json
import os as _qgram_os
import sys as _qgram_sys
import time as _qgram_time
from pathlib import Path as _qgram_Path

_qgram_started = _qgram_time.monotonic()
_qgram_root = _qgram_Path('/kaggle/working')
_qgram_log = (_qgram_root / 'qwengram-remote-events.jsonl').open('w', buffering=1)


def _qgram_event(data, stream='stdout'):
    _qgram_log.write(_qgram_json.dumps({'time':_qgram_time.monotonic()-_qgram_started,
                                     'stream_name':stream, 'data':data})+'\n')


class _QgramTee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, data):
        _qgram_event(data, self.stream)
        return self.original.write(data)

    def flush(self):
        self.original.flush()
        _qgram_log.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def _qgram_save(state):
    tmp = _qgram_root / 'qwengram-cell.tmp'
    with tmp.open('w') as f:
        _qgram_json.dump(state, f)
        f.flush(); _qgram_os.fsync(f.fileno())
    _qgram_os.replace(tmp, _qgram_root / 'qwengram-cell.json')


def _qgram_cell(job, cell, source_sha, code, last=False):
    state = {'job':job, 'cell':cell, 'source_sha256':source_sha, 'status':'running'}
    _qgram_save(state)
    marker = 'CELL_START '+cell+'\n'
    _qgram_event(marker); print(marker, end='', flush=True)
    with _qgram_context.redirect_stdout(_QgramTee(_qgram_sys.stdout, 'stdout')), \
         _qgram_context.redirect_stderr(_QgramTee(_qgram_sys.stderr, 'stderr')):
        result = get_ipython().run_cell(code, store_history=False)
    state['elapsed_s'] = _qgram_time.monotonic()-_qgram_started
    state['status'] = 'done' if result.success else 'error'
    if result.success and last: state['status'] = 'notebook_complete'
    if not result.success:
        _qgram_save(state)
        raise RuntimeError('Notebook cell failed: '+cell)
    marker = 'CELL_DONE '+cell+'\n'
    _qgram_event(marker); print(marker, end='', flush=True)
    if last:
        marker = 'NOTEBOOK_COMPLETE '+job+'\n'
        _qgram_event(marker); print(marker, end='', flush=True)
    _qgram_save(state)
