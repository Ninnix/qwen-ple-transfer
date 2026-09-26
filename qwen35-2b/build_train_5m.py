import json
from pathlib import Path

ROOT = Path(__file__).parent
REF = json.loads((ROOT.parent / 'qwen35-08b/reader-scale-20m/kaggle_reader_20m_canonical.ipynb').read_text())
BENCH = json.loads((ROOT / 'benchmark.ipynb').read_text())
PIN = json.loads((ROOT / 'frozen.json').read_text())
OUT = ROOT / 'train-5m'
OUT.mkdir(exist_ok=True)


def source(nb, name):
    return ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == name))


def cell(name, body):
    return {'cell_type': 'code', 'execution_count': None, 'id': name,
            'metadata': {}, 'outputs': [], 'source': body.splitlines(keepends=True)}


cells = [cell(c['id'], ''.join(c['source'])) for c in BENCH['cells'] if c['id'] != 'c-benchmark']
data = source(REF, 'c-data').replace('/kaggle/working/ple-08b', '/kaggle/working/qwengram-2b')
data = data.replace('self.tok=tok\n', 'self.tok=tok\n        self.documents_read=0\n')
data = data.replace("t=(next(self.it) or {}).get('text') or ''", "t=(next(self.it) or {}).get('text') or ''\n            self.documents_read+=1")
cells.append(cell('c-data', data))

valprep = source(REF, 'c-valprep')
valprep += '''
from pathlib import Path
import sys
STAGE = _hits[0].parent
for name, record in _man.items():
    if name.startswith(('frozen-v1__', 'calib__')) and name.endswith(('.json', '.uint32le')):
        path = STAGE / name
        assert path.stat().st_size == record['size']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
for name in ('code__config.py', 'code__frozen_data.py'):
    path = STAGE / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _man[name]['sha256']
    dest = WORK / name.split('__', 1)[1]
    shutil.copyfile(path, dest)
sys.path.insert(0, str(WORK))
import frozen_data
hellaswag, hs_meta = frozen_data.hellaswag_rows(1000, secret_value_0)
lambada, lam_meta = frozen_data.lambada_rows(1000, secret_value_0)
assert len(hellaswag) == len(lambada) == 1000
EVAL_MANIFEST = {'full': mfull['tokens_sha256'], 'fast': mf['tokens_sha256'],
                 'domains': {d: _man['frozen-v1__tokens-' + d + '.uint32le']['sha256']
                             for d in ('general', 'code', 'math', 'scientific', 'multilingual')},
                 'calibration': _man['calib__calib-750k-blocks.uint32le']['sha256'],
                 'hellaswag': hashlib.sha256(json.dumps(hellaswag, sort_keys=True).encode()).hexdigest(),
                 'lambada': hashlib.sha256(json.dumps(lambada, sort_keys=True).encode()).hexdigest(),
                 'hellaswag_source': hs_meta, 'lambada_source': lam_meta}
(WORK / 'evaluation-streams.json').write_text(json.dumps(EVAL_MANIFEST, indent=2))
print('frozen evaluation streams', json.dumps(EVAL_MANIFEST), flush=True)
'''
cells.append(cell('c-valprep', valprep))

freeze = '''import hashlib, json, os, time
from pathlib import Path
from safetensors.torch import save_file, load_file

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

assert mfull['dataset_rev'] == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
stream = FineWebEduStream(tokenizer, mfull['skip_docs'], mfull['dataset_rev'])
FROZEN_TRAIN = freeze_stream_tokens(tokenizer, stream, 15000000)
assert FROZEN_TRAIN.shape == (29297, 512)
TRAIN_PATH = WORK / 'reader-train-15m.uint32le'
raw = FROZEN_TRAIN.numpy().astype('<u4').tobytes()
TRAIN_SHA = hashlib.sha256(raw).hexdigest()
with TRAIN_PATH.open('wb') as f:
    f.write(raw); f.flush(); os.fsync(f.fileno())
assert sha(TRAIN_PATH) == TRAIN_SHA
TRAIN_MANIFEST = {'dataset': C.DATASET_ID, 'config': C.DATASET_CONFIG,
                  'dataset_revision': mfull['dataset_rev'],
                  'skip_documents': mfull['skip_docs'],
                  'first_training_document': mfull['skip_docs'],
                  'documents_read': stream.documents_read,
                  'last_training_document_exclusive': mfull['skip_docs'] + stream.documents_read,
                  'token_count': int(FROZEN_TRAIN.numel()), 'seq': C.SEQ,
                  'sha256': TRAIN_SHA, 'eos': C.EOS,
                  'tokenizer_sha256': '%s', 'target_revision': '%s'}
(WORK / 'reader-train-15m.json').write_text(json.dumps(TRAIN_MANIFEST, indent=2))
TRAIN_MANIFEST_SHA = sha(WORK / 'reader-train-15m.json')
print('frozen reader stream', json.dumps(TRAIN_MANIFEST), flush=True)
''' % (PIN['target_tokenizer_sha256'], PIN['target_revision'])
cells.append(cell('c-freeze-train', freeze))

tests = source(REF, 'c-tests')
tests = tests.replace("print('ALL CPU-capable correctness gates PASSED (mount gate informational))')",
                      "assert len(find_ple_manifests()) == 11\nprint('correctness gates passed')")
cells.append(cell('c-tests', tests))

cache = source(REF, 'c-compact')
cache = cache.replace("    _stream=FineWebEduStream(tokenizer, _mfF['skip_docs'], _mfF['dataset_rev'])\n", '')
cache = cache.replace("_train_toks=freeze_stream_tokens(tokenizer, _stream, 20000000)[15000064//C.SEQ:]",
                      '_train_toks=FROZEN_TRAIN[:5000192//C.SEQ]')
cache = cache.replace('compact-reader-suffix', 'compact-reader-0-5m')
cache = cache.replace("'train_start':15000064,'train_end':int(20000000)",
                      "'train_start':0,'train_end':5000192")
cache = cache.replace("'target_rev':_mfF.get('target_rev')", "'target_rev':'%s'" % PIN['target_revision'])
cache = cache.replace("           'train_start':0", "           'full_train_sha256':TRAIN_SHA,'train_manifest_sha256':TRAIN_MANIFEST_SHA,\n           'train_start':0")
cache += '''
assert _cstore is not None and _cmp is not None
CACHE_META = json.loads((_cdir / 'compact.json').read_text())
CACHE_SHA = CACHE_META['rows_sha256']
ADDR_SHA = CACHE_META['addrs_sha256']
assert CACHE_META['ple_revision'] == '%s'
shutil.copyfile(_cdir / 'compact.json', WORK / 'compact-reader-0-5m.json')
print('cache row SHA', CACHE_SHA, 'mapping SHA', ADDR_SHA, flush=True)
''' % PIN['source_revision']
cells.append(cell('c-compact', cache))

trainer = source(REF, 'c-trainfn')
start = trainer.index('    _vmeta=json.loads((VALDIR/')
end = trainer.index('    full_set=set(full_at)', start)
trainer = trainer[:start] + '''    chunks=FROZEN_TRAIN[:math.ceil(max_tokens/C.SEQ)]
''' + trainer[end:]
start = trainer.index('    import hashlib\n    assert store.meta')
end = trainer.index('    for step, _ids', start)
trainer = trainer[:start] + '''    assert store.meta['train_start'] <= _seen0
    assert store.meta['train_end'] >= math.ceil(max_tokens/C.SEQ)*C.SEQ
    assert torch.equal(chunks, FROZEN_TRAIN[_skip:_skip+len(chunks)])
    print('frozen training stream SHA verified', TRAIN_SHA, flush=True)
''' + trainer[end:]
trainer = trainer.replace("'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}",
                          "'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),\n"
                          "                    'python_rng':random.getstate(),'optimizer_step':step,\n"
                          "                    'provenance':RUN_META}")
trainer = trainer.replace("torch.save(_ckpt, f'/kaggle/working/reader-{tag}-{seen}.pt')",
                          "atomic_checkpoint(_ckpt, WORK / ('reader-%s-%d.pt' % (tag, seen)))")
trainer = trainer.replace('    opt=AdamW(tr,lr=C.LR,weight_decay=C.WD)\n',
                          "    opt=AdamW(tr,lr=C.LR,weight_decay=C.WD)\n"
                          "    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in inj.parameters()}\n")
cells.append(cell('c-trainfn', trainer))

run = '''import copy, math, os, random

def atomic_checkpoint(state, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        torch.save(state, f)
        f.flush(); os.fsync(f.fileno())
    test = torch.load(tmp, map_location='cpu', weights_only=False)
    assert test['seen'] == state['seen'] and test['optimizer_step'] == state['optimizer_step']
    assert test['provenance'] == state['provenance']
    assert all(torch.equal(test['reader'][k], v) for k, v in state['reader'].items())
    os.replace(tmp, path)
    print('CHECKPOINT', path.name, sha(path), flush=True)

def pre_eval_save_reader(inj, seen, threshold):
    path = WORK / ('reader-%d.safetensors' % seen)
    tmp = path.with_suffix('.safetensors.tmp')
    state = {k: v.detach().cpu() for k, v in inj.state_dict().items()}
    save_file(state, str(tmp))
    with tmp.open('rb+') as f: f.flush(); os.fsync(f.fileno())
    check = load_file(str(tmp), device='cpu')
    assert set(check) == set(state)
    assert all(torch.equal(check[k], v) for k, v in state.items())
    os.replace(tmp, path)
    print('READER', threshold, seen, sha(path), flush=True)

RUN_META = {'target_revision': '__TARGET_REV__', 'tokenizer_sha256': '__TOKENIZER_SHA__',
            'ple_revision': '__PLE_REV__', 'train_stream_sha256': TRAIN_SHA,
            'train_manifest_sha256': TRAIN_MANIFEST_SHA,
            'eval_streams_sha256': sha(WORK / 'evaluation-streams.json'),
            'compact_cache_sha256': CACHE_SHA, 'address_mapping_sha256': ADDR_SHA,
            'injection_idx': [2, 8], 'human_layers': [3, 9],
            'reader_branches': 1, 'hidden': 2048,
            'lr': C.LR, 'weight_decay': C.WD, 'gamma_init': C.GAMMA_INIT}
assert not any(p.requires_grad for p in model.parameters())
assert _cmp.meta['train_tokens_sha256'] == hashlib.sha256(FROZEN_TRAIN[:9766].numpy().astype('<u4').tobytes()).hexdigest()
assert all(p.stat().st_size > 0 for p in _cstore.part_paths.values())
PLE_STATS = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
base = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, 0.0).to('cuda')
stock = eval_loss(base, None, val_fast, False)
base.close()
baselines = {'fast': stock}
print('stock fast validation', stock, flush=True)

one = train_reader([2, 8], 1, 1000000, _cmp, val_fast, 'real-1m',
                   ckpts=[1000000], baselines=baselines)
path = WORK / ('reader-real-1m-%d.pt' % one['checkpoints'][-1]['tokens'])
resume = torch.load(path, map_location='cpu', weights_only=False)
assert resume['seen'] == 1000448
RUN_META['schedule'] = {'mode': 'continuation', 'peak': 1.5e-5,
                        'warm_steps': 200, 'horizon_tokens': 5000000}
five = train_reader([2, 8], 1, 5000000, _cmp, val_fast, 'real-5m',
                    ckpts=[5000000], baselines=baselines,
                    resume=resume, cont_peak=1.5e-5, cont_warm=200)
seen = five['checkpoints'][-1]['tokens']
assert seen == 5000192
ck = WORK / ('reader-real-5m-%d.pt' % seen)
verified = torch.load(ck, map_location='cpu', weights_only=False)
assert verified['seen'] == seen and verified['provenance']['train_stream_sha256'] == TRAIN_SHA
assert PLE_STATS == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
assert sha(WORK / ('reader-%d.safetensors' % seen))
summary = {'stock_fast': stock, 'one_m': one['checkpoints'][-1],
           'five_m': five['checkpoints'][-1], 'reader_sha256': sha(WORK / ('reader-%d.safetensors' % seen)),
           'checkpoint_sha256': sha(ck), 'train_sha256': TRAIN_SHA,
           'cache_sha256': CACHE_SHA, 'address_sha256': ADDR_SHA}
(WORK / 'reader-5m-summary.json').write_text(json.dumps(summary, indent=2))
print('READER_5M_COMPLETE', json.dumps(summary), flush=True)
'''
run = run.replace('__TARGET_REV__', PIN['target_revision']).replace('__TOKENIZER_SHA__', PIN['target_tokenizer_sha256']).replace('__PLE_REV__', PIN['source_revision'])
cells.append(cell('c-train-5m', run))

nb = {'cells': cells, 'metadata': BENCH['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
(OUT / 'train_5m.ipynb').write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / 'kernel-metadata.json').read_text())
meta.update(code_file='train_5m.ipynb',
            id='ninnix/qwengram-2b-reader-0-5m-t4',
            title='Qwengram 2B reader 0 5M T4',
            dataset_sources=['ninnix/qwen38-ple-p%02d' % i for i in range(11)] +
                            ['ninnix/qwen-ple-arb-cont-750k-stage'])
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
