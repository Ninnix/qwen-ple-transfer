import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent / 'qwen35-2b'
PIN = json.loads((HERE / 'frozen.json').read_text())
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
BASE = json.loads((REF / 'train-5m/train_5m.ipynb').read_text())
EXECUTION = json.loads((HERE / 'execution.json').read_text()) if (HERE / 'execution.json').exists() else {'strategy':'single-T4'}
MULTI = EXECUTION['strategy'] == 'contiguous-T4x2'
CHECKPOINTED = EXECUTION['strategy'] == 'single-T4-checkpointed'
EXECUTION_CODE = 'model_parallel.py' if MULTI else 'checkpointing.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(nb, name):
    return ''.join(next(c['source'] for c in nb['cells'] if c['id'] == name))


def adapt(s):
    replacements = {
        'Qwen3.5-2B': 'Qwen3.5-4B', 'qwengram-2b': 'qwengram-4b',
        'QWENGRAM_2B': 'QWENGRAM_4B', 'Qwengram 2B': 'Qwengram 4B',
        '15852e8c16360a2fea060d615a32b45270f8a8fc': PIN['target_revision'],
        'HIDDEN: int = 2048': 'HIDDEN: int = 2560', 'N_LAYERS: int = 24': 'N_LAYERS: int = 32',
        'hidden=2048': 'hidden=2560', 'hidden_size==2048': 'hidden_size==2560',
        'num_hidden_layers==24': 'num_hidden_layers==32', 'len(o)==24': 'len(o)==32',
        'torch.randn(1,4,2048)': 'torch.randn(1,4,2560)',
        "'hidden': 2048": "'hidden': 2560", '(1, 1, 2048,': '(1, 1, 2560,',
        'arb.w.numel() == 2048': 'arb.w.numel() == 2560', '== 2050': '== 2562',
        '[2, 8]': '[3, 11]', '(2, 8)': '(3, 11)', '(2,8)': '(3,11)',
        '[3, 9]': '[4, 12]', "('2', '8')": "('3', '11')",
        "GAMMAS['2']": "GAMMAS['3']", "GAMMAS['8']": "GAMMAS['11']",
        'site == 2': 'site == 3', 'site == 8': 'site == 11',
        'frozen.readers["2"]': 'frozen.readers["3"]',
        'frozen.readers["8"]': 'frozen.readers["11"]',
        'dec[2].register': 'dec[3].register', 'dec[8].register': 'dec[11].register',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def cell(name, body):
    code = '\n'.join(line for line in body.splitlines() if not line.startswith('%'))
    ast.parse(code, filename=name)
    return {'cell_type': 'code', 'execution_count': None, 'id': name,
            'metadata': {}, 'outputs': [], 'source': body.splitlines(keepends=True)}


def save(name, cells, slug):
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    if MULTI or CHECKPOINTED:
        index = next(i for i,c in enumerate(cells) if c['id']=='c-load')
        cells.insert(index+1, cell('c-execution-parity', execution_cell()))
    nb = {'cells': cells, 'metadata': BASE['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
    (folder / (name + '.ipynb')).write_text(json.dumps(nb, indent=1) + '\n')
    meta = json.loads((REF / 'kernel-metadata.json').read_text())
    meta.update(code_file=name + '.ipynb', id='ninnix/' + slug, title=slug.replace('-', ' '),
                dataset_sources=['ninnix/qwen38-ple-p%02d' % i for i in range(11)] +
                                ['ninnix/qwen-ple-arb-cont-750k-stage', 'ninnix/qwengram-4b-checkpoints'])
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


def execution_cell():
    setup = 'ContiguousT4(model,16)' if MULTI else 'CheckpointedDecoder(model.model.layers)'
    body = (HERE / EXECUTION_CODE).read_text() + '''
import gc
_mp_ids = val_full[:1,:64].cuda()
_mp_memory = _cmp.lookup(addresses(_mp_ids.cpu())).cuda()
with torch.inference_mode():
    _mp_stock = model(input_ids=_mp_ids, use_cache=False).logits.detach().cpu()
torch.manual_seed(1234)
_mp_probe = ReaderInjection(model,SITES,C.MEM_DIM,C.HIDDEN,1,C.GAMMA_INIT).cuda()
_mp_probe.set_memory(_mp_memory)
_mp_state = {k:v.detach().cpu().clone() for k,v in _mp_probe.state_dict().items()}
_mp_logits = model(input_ids=_mp_ids,use_cache=False).logits
torch.nn.functional.cross_entropy(_mp_logits[:,:-1].float().reshape(-1,C.VOCAB),_mp_ids[:,1:].reshape(-1)).backward()
_mp_expected = _mp_logits.detach().cpu()
_mp_grads = {k:p.grad.detach().cpu().clone() for k,p in _mp_probe.named_parameters()}
_mp_probe.close()
del _mp_probe,_mp_logits
gc.collect(); torch.cuda.empty_cache()
EXECUTOR = __SETUP__
with torch.inference_mode():
    _mp_actual_stock = model(input_ids=_mp_ids,use_cache=False).logits.detach().cpu()
_mp_stock_diff = (_mp_stock.float()-_mp_actual_stock.float()).abs().max().item()
assert _mp_stock_diff <= 0.002
_mp_probe = ReaderInjection(model,SITES,C.MEM_DIM,C.HIDDEN,1,C.GAMMA_INIT).cuda()
_mp_probe.load_state_dict(_mp_state); _mp_probe.set_memory(_mp_memory)
_mp_logits = model(input_ids=_mp_ids,use_cache=False).logits
torch.nn.functional.cross_entropy(_mp_logits[:,:-1].float().reshape(-1,C.VOCAB),_mp_ids[:,1:].reshape(-1)).backward()
_mp_reader_diff = (_mp_expected.float()-_mp_logits.detach().float().cpu()).abs().max().item()
assert _mp_reader_diff <= 0.002
_mp_grad_diff = {}
for name,p in _mp_probe.named_parameters():
    torch.testing.assert_close(p.grad.cpu(),_mp_grads[name],atol=1e-6,rtol=1e-4)
    _mp_grad_diff[name] = (p.grad.cpu()-_mp_grads[name]).abs().max().item()
assert all(not p.requires_grad and p.grad is None for p in model.parameters())
if EXECUTION['strategy'] == 'single-T4-checkpointed':
    assert all(p.device.index == 0 for p in model.parameters())
    assert EXECUTOR.body_calls > EXECUTOR.checkpoint_calls > 0
_mp_probe.close()
MP_PARITY = {**EXECUTOR.description(),'stock_max_abs_logit':_mp_stock_diff,
             'reader_max_abs_logit':_mp_reader_diff,'reader_gradient_max_abs':_mp_grad_diff,
             'logit_atol':0.002,'gradient_atol':1e-6,'gradient_rtol':1e-4}
(WORK/'execution-parity.json').write_text(json.dumps(MP_PARITY,indent=2))
print('EXECUTION_PARITY',json.dumps(MP_PARITY),flush=True)
del p,_mp_probe,_mp_state,_mp_grads,_mp_logits,_mp_ids,_mp_memory,_mp_expected,_mp_stock,_mp_actual_stock
if hasattr(EXECUTOR,'cache'): EXECUTOR.cache.clear()
gc.collect()
for i in GPU_IDS:
    with torch.cuda.device(i): torch.cuda.empty_cache()
'''
    return body.replace('__SETUP__',setup)


def shared():
    cells = []
    for name in ('c-env', 'c-install', 'c-config', 'c-hashing', 'c-reader', 'c-inject', 'c-ple', 'c-tok', 'c-data'):
        s = adapt(source(BASE, name))
        if name == 'c-env' and CHECKPOINTED:
            s = s.replace('import os, shutil, sys', "import os, shutil, sys\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'")
        if name == 'c-config':
            s = '\n'.join(line.split('#', 1)[0].rstrip() for line in s.splitlines()) + '\n'
            s += '\nSITES = (3, 11)\nassert C.HIDDEN == 2560 and C.N_LAYERS == 32 and C.PLACEMENTS == (SITES,)\n'
            s += '\nEXECUTION = %r\nGPU_IDS = %r\n' % (EXECUTION, (0, 1) if MULTI else (0,))
            s += '''
def gpu_sync():
    for i in GPU_IDS: torch.cuda.synchronize(i)

def gpu_reset():
    for i in GPU_IDS: torch.cuda.reset_peak_memory_stats(i)

def gpu_peaks():
    return [torch.cuda.max_memory_allocated(i)/1024**3 for i in GPU_IDS]
'''
        if name == 'c-tok':
            s = s.replace('assert api.model_info(C.TARGET_ID).sha == trev', '')
            s += '\ntokenizer = tt\n'
        if name == 'c-data':
            a = s.index('    from huggingface_hub import HfApi\n')
            b = s.index('\ndef load_validation', a)
            s = s[:a] + '''    paths = [VALDIR / ('validation-%s.json' % name) for name in ('fast', 'full')]
    assert all(p.exists() for p in paths), 'Frozen validation artifacts missing'
    return tuple(json.loads(p.read_text()) for p in paths)
''' + s[b:]
        cells.append(cell(name, s))
    val = adapt(source(BASE, 'c-valprep'))
    val += '\nassert EVAL_MANIFEST == ' + repr(json.loads((REF / 'results/evaluation-streams.json').read_text())) + '\n'
    cells.append(cell('c-valprep', val))
    freeze = '''import hashlib, os, random, shutil, time
from array import array
from safetensors.torch import save_file, load_file

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

_train_hits = list(Path('/kaggle/input').rglob('reader-train-15m.uint32le'))
assert len(_train_hits) == 1
TRAIN_PATH = _train_hits[0]
TRAIN_MANIFEST_PATH = TRAIN_PATH.with_suffix('.json')
TRAIN_MANIFEST = json.loads(TRAIN_MANIFEST_PATH.read_text())
TRAIN_SHA = sha(TRAIN_PATH)
TRAIN_MANIFEST_SHA = sha(TRAIN_MANIFEST_PATH)
assert TRAIN_SHA == __TRAIN_SHA__ == TRAIN_MANIFEST['sha256']
assert TRAIN_MANIFEST_SHA == __MANIFEST_SHA__
assert TRAIN_MANIFEST['tokenizer_sha256'] == __TOKENIZER_SHA__
assert TRAIN_MANIFEST['token_count'] == 15000064
_tokens = array('I'); _tokens.frombytes(TRAIN_PATH.read_bytes())
FROZEN_TRAIN = torch.tensor(_tokens, dtype=torch.long).view(-1, 512)
assert FROZEN_TRAIN.shape == (29297, 512)
del _tokens
shutil.copyfile(TRAIN_MANIFEST_PATH, WORK / 'reader-train-15m.json')
print('Reused exact 2B frozen stream', TRAIN_SHA, TRAIN_MANIFEST_SHA, flush=True)
'''.replace('__TRAIN_SHA__', repr(PROTOCOL['train_stream_sha256'])).replace('__MANIFEST_SHA__', repr(PROTOCOL['train_manifest_sha256'])).replace('__TOKENIZER_SHA__', repr(PIN['target_tokenizer_sha256']))
    cells.append(cell('c-freeze-train', freeze))
    return cells


def preflight():
    return '''import gc, json, time, platform, importlib.metadata
import torch.nn.functional as F

RUNTIME = {'python':platform.python_version(), 'torch':torch.__version__,
           'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
           'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
           'packages':{k:importlib.metadata.version(k) for k in ('transformers','safetensors','accelerate','tokenizers')},
           'gpus':[{'name':torch.cuda.get_device_name(i),
                    'total_bytes':torch.cuda.get_device_properties(i).total_memory,
                    'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]}
(WORK / 'runtime.json').write_text(json.dumps(RUNTIME, indent=2))
assert C.HIDDEN == 2560 and C.N_LAYERS == 32 and SITES == (3, 11)
assert all(not p.requires_grad and p.grad is None for p in model.parameters())
assert torch.cuda.get_device_capability(0) == (7, 5)
if EXECUTION['strategy'] == 'single-T4-checkpointed': assert torch.cuda.device_count() == 1
assert ACTIVE_DTYPE == torch.float16
assert all(next(decoder_layers(model)[i].parameters()).device.index == 0 for i in SITES)
assert all(decoder_layers(model)[i].block_type == 'full_attention' for i in SITES)
assert _cstore.ple_revision == '__PLE_REV__'
assert all(Path(p).is_relative_to('/kaggle/input') for p in _cstore.part_paths.values())
PLE_STATS = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
BACKBONE_VERSIONS = {n: p._version for n, p in model.named_parameters()}
_qual_start = int(_cmp.meta['train_start']) // C.SEQ
_qual_ids = FROZEN_TRAIN[_qual_start:_qual_start+1, :64].to('cuda')
_qual_mem = _cmp.lookup(addresses(_qual_ids.cpu())).to('cuda')
with torch.inference_mode():
    _stock = model(input_ids=_qual_ids, use_cache=False).logits
_probe = ReaderInjection(model, SITES, C.MEM_DIM, C.HIDDEN, 1, 0.0).to('cuda')
assert _probe.idx == SITES
with torch.inference_mode():
    _probe.set_memory(None)
    _disabled = model(input_ids=_qual_ids, use_cache=False).logits
    _disabled_diff = (_stock.float() - _disabled.float()).abs().max().item()
    _probe.set_memory(_qual_mem)
    _zero = model(input_ids=_qual_ids, use_cache=False).logits
    _zero_diff = (_stock.float() - _zero.float()).abs().max().item()
assert _disabled_diff <= 0.002 and _zero_diff <= 0.002
with torch.no_grad():
    for reader in _probe.readers.values(): reader.gamma.fill_(C.GAMMA_INIT)
    _near = model(input_ids=_qual_ids, use_cache=False).logits
assert torch.isfinite(_near).all()
_near_diff = (_stock.float() - _near.float()).abs().max().item()
_params = list(_probe.parameters())
assert len(_params) == 8 and all(p.dtype == torch.float32 for p in _params)
_opt = torch.optim.AdamW(_params, lr=C.LR, weight_decay=C.WD)
assert {id(p) for g in _opt.param_groups for p in g['params']} == {id(p) for p in _params}
assert not _opt.state
assert not {id(p) for p in _params} & {id(p) for p in model.parameters()}
_probe.close()
QUALIFICATION = {'passed_before_optimizer_step': True, 'disabled_max_abs_logit': _disabled_diff,
                 'zero_gamma_max_abs_logit': _zero_diff, 'near_init_max_abs_logit': _near_diff,
                 'injection_idx': list(SITES), 'train_sha256': TRAIN_SHA,
                 'eval_manifest_sha256': sha(WORK / 'evaluation-streams.json'),
                 'cache_sha256': CACHE_SHA, 'mapping_sha256': ADDR_SHA,
                 'lookup_max_abs_difference': _d, 'optimizer_tensor_count': len(_params),
                 'dtype': str(ACTIVE_DTYPE), 'target_revision': '__TARGET_REV__'}
QUALIFICATION['execution'] = EXECUTION
QUALIFICATION['execution_parity_sha256'] = sha(WORK/'execution-parity.json')
(WORK / 'qualification.json').write_text(json.dumps(QUALIFICATION, indent=2))
del reader, _probe, _params, _opt, _stock, _disabled, _zero, _near, _qual_ids, _qual_mem
gc.collect(); torch.cuda.empty_cache()
print('QUALIFICATION_PASSED', json.dumps(QUALIFICATION), flush=True)
'''.replace('__PLE_REV__', PIN['source_revision']).replace('__TARGET_REV__', PIN['target_revision'])


def benchmark():
    cells = shared()
    cache = adapt(source(BASE, 'c-compact'))
    cache = cache.replace('FROZEN_TRAIN[:5000192//C.SEQ]', 'FROZEN_TRAIN[:8]')
    cache = cache.replace("'train_end':5000192", "'train_end':4096")
    cache = cache.replace("_val,_=load_validation('full')", "_val,_=load_validation('full')\n    _val=_val[:8]")
    cache = cache.replace('compact-reader-0-5m', 'compact-qualification')
    cells.append(cell('c-compact', cache))
    cells.append(cell('c-load', adapt(source(BASE, 'c-load'))))
    cells.append(cell('c-qualification', preflight()))
    bench = '''torch.manual_seed(1234)
reader = ReaderInjection(model, SITES, C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
params = list(reader.parameters())
opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=C.WD)
assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in params}
assert not opt.state
gpu_reset()
with torch.inference_mode():
    for i in range(2):
        ids = FROZEN_TRAIN[i:i+1].cuda()
        reader.set_memory(_cmp.lookup(addresses(ids.cpu())).cuda())
        model(input_ids=ids, use_cache=False)
    gpu_sync()
    start = time.perf_counter()
    for i in range(4):
        ids = FROZEN_TRAIN[i:i+1].cuda()
        reader.set_memory(_cmp.lookup(addresses(ids.cpu())).cuda())
        model(input_ids=ids, use_cache=False)
    gpu_sync()
    forward_s = time.perf_counter() - start
forward_peaks = gpu_peaks()
gpu_reset()
start = time.perf_counter()
for i in range(12):
    ids = FROZEN_TRAIN[i % 8:i % 8 + 1].cuda()
    reader.set_memory(_cmp.lookup(addresses(ids.cpu())).cuda())
    opt.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, use_cache=False).logits
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, C.VOCAB), ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    assert set(opt.state) == set(params)
gpu_sync()
train_s = time.perf_counter() - start
peaks = gpu_peaks()
reserved = [torch.cuda.max_memory_reserved(i)/1024**3 for i in GPU_IDS]
totals = [torch.cuda.get_device_properties(i).total_memory/1024**3 for i in GPU_IDS]
headroom = [total-peak for total,peak in zip(totals,peaks)]
transfer = None
if EXECUTION['strategy'] == 'contiguous-T4x2':
    opt.zero_grad(set_to_none=True)
    del logits,loss
    EXECUTOR.profile = True; EXECUTOR.transfers.clear()
    ids = FROZEN_TRAIN[:1].cuda()
    reader.set_memory(_cmp.lookup(addresses(ids.cpu())).cuda())
    logits = model(input_ids=ids,use_cache=False).logits
    loss = F.cross_entropy(logits[:,:-1].float().reshape(-1,C.VOCAB),ids[:,1:].reshape(-1))
    loss.backward(); gpu_sync()
    EXECUTOR.profile = False
    seconds = sum(t['seconds'] for t in EXECUTOR.transfers)
    transfer = {'events':EXECUTOR.transfers.copy(), 'total_seconds':seconds,
                'bytes':sum(t['bytes'] for t in EXECUTOR.transfers),
                'fraction_of_normal_step_estimate':seconds/(train_s/12),
                'timing':'Separate synchronized forward/backward; no optimizer step; excludes pre-copy synchronization'}
report = {**QUALIFICATION, 'forward_tok_s': 4*512/forward_s, 'training_tok_s': 12*512/train_s,
          'forward_peaks_gib': forward_peaks, 'training_peak_gib': max(peaks),
          'gpu_peaks_gib':peaks, 'gpu_reserved_peaks_gib':reserved,
          'gpu_totals_gib':totals, 'gpu_headroom_gib':headroom,
          'inter_gpu_transfer':transfer,
          'host_rss_mib': rss_mb(), 'cache_bytes': (_cdir/'rows.u8').stat().st_size+(_cdir/'addrs.u32').stat().st_size,
          'training_wall_s': train_s, 'reader_parameters': sum(p.numel() for p in params),
          'loss': float(loss.detach()), 'benchmark_reader_discarded': True,
          'configuration_qualified': min(headroom) >= 0.5 and 12*512/train_s >= 150}
report['single_t4_qualified'] = EXECUTION['strategy'].startswith('single-T4') and report['configuration_qualified']
report['execution_details'] = EXECUTOR.description()
assert all(p.device.index == 0 for p in model.parameters()) if report['single_t4_qualified'] else True
assert PLE_STATS == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
assert BACKBONE_VERSIONS == {n: p._version for n, p in model.named_parameters()}
(WORK / 'benchmark-4b.json').write_text(json.dumps(report, indent=2))
print('BENCHMARK_4B', json.dumps(report), flush=True)
reader.close()
assert report['configuration_qualified'], 'Execution needs infrastructure review'
'''
    cells.append(cell('c-benchmark', bench))
    slug = 'qwengram-4b-t4x2-qualification' if MULTI else 'qwengram-4b-single-t4-checkpointed'
    save('benchmark', cells, slug)


def reader(stage):
    ref = json.loads((REF / ('train-%dm/train_%dm.ipynb' % (stage, stage))).read_text())
    cells = shared()
    cache = adapt(source(ref, 'c-compact'))
    cache = cache.replace('one cache serves IDX 2, 8, 2+8: placement never changes addresses',
                          'same frozen addressing at IDX3 and IDX11')
    rows, mapping = {
        5: ('a355f46fcc7d0ea9b76dee03955acff0948584a1432752ce5ed4856fa3308219',
            'b2a01c076a21f46b3a7b24e386c143a90745291d054a66e7e61fc9eef13559d9'),
        10: ('337b79203d5f74436b23470964c68154ab6cc73ed38f7fc8b3fa30ea0fd9da61',
             '350908df637fd8fa4c62ddeed09dcfada54eec4fbf2ed3d2a2f31c18aa2639d7'),
        15: ('7f1cd946387ecf2a8399ded9530353c613a8d0bcef1de9a270e0011fef2bcb0e',
             'a197580d91b2eb224e45961d6dc83f78233bc430ffbe59f553ed6534866f3b60'),
    }[stage]
    cache += '\nassert CACHE_SHA == %r and ADDR_SHA == %r\n' % (rows, mapping)
    cells.append(cell('c-compact', cache))
    cells.append(cell('c-load', adapt(source(BASE, 'c-load'))))
    cells.append(cell('c-qualification', preflight()))
    cells.append(cell('c-benchmark-gate', '''_bench_hits = list(Path('/kaggle/input').rglob('benchmark-4b.json'))
assert len(_bench_hits) == 1, 'Persist qualification before real training'
BENCHMARK = json.loads(_bench_hits[0].read_text())
assert BENCHMARK['configuration_qualified'] and BENCHMARK['passed_before_optimizer_step']
assert BENCHMARK['execution'] == EXECUTION
assert BENCHMARK['target_revision'] == trev
assert BENCHMARK['injection_idx'] == list(SITES)
assert BENCHMARK['train_sha256'] == TRAIN_SHA
assert BENCHMARK['lookup_max_abs_difference'] == 0
'''))
    trainer = adapt(source(ref, 'c-trainfn'))
    trainer = trainer[:trainer.index('\ndef write_bundle')]
    trainer = trainer.replace("torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])",
                              "torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])\n        random.setstate(resume['python_rng'])")
    trainer = trainer.replace("'non-finite loss — fallback to FP32 and rerun'", "'non-finite loss: abort this run'")
    trainer = trainer.replace('clip_grad_norm_(inj.parameters(),1.0)',
                              'clip_grad_norm_(inj.parameters(),1.0,error_if_nonfinite=True)')
    trainer = trainer.replace("'tok_s':round(seen/(time.perf_counter()-t0),1)",
                              "'tok_s':round((seen-_seen0)/(time.perf_counter()-t0),1), 'wall_s':time.perf_counter()-t0")
    trainer = trainer.replace("'peak_GiB':round(torch.cuda.max_memory_allocated()/1024**3,3)",
                              "'peak_GiB':round(max(gpu_peaks()),3), 'gpu_peaks_gib':gpu_peaks()")
    trainer = trainer.replace("'python_rng':random.getstate(),'optimizer_step':step,",
                              "'python_rng':random.getstate(),'optimizer_step':step,\n                    'corpus_cursor': {'next_block':step, 'tokens':seen, 'stream_sha256':TRAIN_SHA},")
    trainer = trainer.replace("    print('frozen training stream SHA verified', TRAIN_SHA, flush=True)",
                              """    assert tuple(layers) == SITES and branches == 1
    assert len(tr) == 8 and all(p.dtype == torch.float32 for p in tr)
    assert not set(tr) & set(model.parameters())
    if resume is None: assert not opt.state
    else:
        assert set(opt.state) == set(tr)
        assert resume['corpus_cursor'] == {'next_block':_skip, 'tokens':_seen0, 'stream_sha256':TRAIN_SHA}
    gpu_reset()
    print('frozen training stream SHA verified', TRAIN_SHA, flush=True)""")
    trainer = trainer.replace('        while pending and seen>=pending[0]:', """        if step % 100 == 0:
            print('TRAIN_PROGRESS', json.dumps({'step':step, 'tokens':seen, 'loss':loss.item(),
                  'lr':opt.param_groups[0]['lr'], 'tok_s':(seen-_seen0)/(time.perf_counter()-t0),
                  'gpu_peaks_gib':gpu_peaks(), 'host_rss_mib':rss_mb()}), flush=True)
        while pending and seen>=pending[0]:""")
    trainer = trainer.replace('            pre_eval_save_reader(inj,seen,th)', """            assert BACKBONE_VERSIONS == {n:p._version for n,p in model.named_parameters()}
            assert PLE_STATS == {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
            assert set(opt.state) == set(tr)
            pre_eval_save_reader(inj,seen,th)""")
    cells.append(cell('c-trainfn', trainer))
    run = adapt(source(ref, 'c-train-5m'))
    run = run.replace("    assert test['seen'] == state['seen'] and test['optimizer_step'] == state['optimizer_step']\n    assert test['provenance'] == state['provenance']\n    assert all(torch.equal(test['reader'][k], v) for k, v in state['reader'].items())",
                      '    assert same_state(test, state)')
    run = run.replace('    os.replace(tmp, path)', '    os.replace(tmp, path)\n    fd = os.open(path.parent, os.O_RDONLY)\n    try: os.fsync(fd)\n    finally: os.close(fd)')
    insert = '''def same_state(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b

'''
    run = run.replace('def atomic_checkpoint', insert + 'def atomic_checkpoint')
    run = run.replace('assert not any(p.requires_grad for p in model.parameters())',
                      "RUN_META.update(%r)\nassert not any(p.requires_grad for p in model.parameters())" % {
                          'model_manifest_sha256': PROTOCOL['model_manifest_sha256'],
                          'protocol_sha256': sha((HERE / 'protocol.json').read_bytes()),
                          'ple_manifest_sha256': PIN['ple_manifest_sha256'],
                          'tokenizer_revision': PIN['target_revision'],
                          'relative_depth': [0.125, 0.375], 'execution': EXECUTION,
                          'execution_sha256': sha((HERE / 'execution.json').read_bytes()),
                          'execution_code_sha256': sha((HERE / EXECUTION_CODE).read_bytes())})
    run += "\nassert BACKBONE_VERSIONS == {n:p._version for n,p in model.named_parameters()}\n"
    if stage == 5:
        first = run.index('one = train_reader(')
        end = run.index("RUN_META['schedule']", first)
        original = run[first:end]
        recovery = '''_recovery = list(Path('/kaggle/input').rglob('recovery-1m.json'))
if _recovery:
    assert len(_recovery) == 1
    recovery = json.loads(_recovery[0].read_text())
    path = _recovery[0].parent / 'reader-real-1m-1000448.pt'
    assert sha(path) == recovery['checkpoint_sha256'] == '5d8aa62b5341fc9ed9e65ad3be81981d551a7e4cc212d3b7da76755a601c0fa0'
    resume = torch.load(path, map_location='cpu', weights_only=False)
    assert resume['seen'] == 1000448 and resume['optimizer_step'] == 1954
    assert resume['provenance']['target_revision'] == RUN_META['target_revision']
    assert resume['provenance']['train_stream_sha256'] == TRAIN_SHA
    assert resume['provenance']['compact_cache_sha256'] == CACHE_SHA
    assert resume['provenance']['execution'] == EXECUTION
    assert len(resume['optimizer']['state']) == len(resume['reader']) == 8
    one = {'checkpoints':[recovery['one_m']]}
    print('EXACT_RECOVERY', resume['seen'], sha(path), flush=True)
else:
'''
        run = run[:first] + recovery + ''.join('    '+line+'\n' for line in original.splitlines()) + run[end:]
    cells.append(cell('c-train-stage', run))
    save('train-%dm' % stage, cells, 'qwengram-4b-reader-%d-%dm-t4' % (stage-5, stage))


def arbitration(stage):
    ref = json.loads((REF / 'arb-eval/arb_eval.ipynb').read_text())
    cells = shared()
    extra = source(ref, 'c-valprep').split("for name in ('code__evaluate.py', 'code__bootstrap.py'):", 1)[1]
    cells.append(cell('c-eval-code', "for name in ('code__evaluate.py', 'code__bootstrap.py'):" + extra))
    cache = adapt(source(ref, 'c-cache-arb'))
    cache += '''
_cmp = store; _cstore = master; _d = maxdiff
CACHE_SHA = cache_meta['rows_sha256']; ADDR_SHA = cache_meta['addrs_sha256']
shutil.copyfile(cache_dir / 'compact.json', WORK / 'compact-arbitration.json')
'''
    cells.append(cell('c-cache-arb', cache))
    cells.append(cell('c-load', adapt(source(BASE, 'c-load'))))
    qualification = preflight()
    qualification = qualification.replace("_qual_start = int(_cmp.meta['train_start']) // C.SEQ\n_qual_ids = FROZEN_TRAIN[_qual_start:_qual_start+1, :64].to('cuda')",
                                          "_qual_ids = val_full[:1, :64].to('cuda')")
    cells.append(cell('c-qualification', qualification))
    for name in ('c-reader-load', 'c-arb-math', 'c-train-arb', 'c-evaluation'):
        s = adapt(source(ref, name))
        if stage == 10:
            s = s.replace('reader-15m-summary', 'reader-10m-summary')
            s = s.replace('reader-real-15m-', 'reader-real-10m-')
            s = s.replace('15000064', '10000384').replace('raw-15m', 'raw-10m')
            s = s.replace('15M frozen reader', '10M frozen reader')
        if name == 'c-reader-load':
            s += "\nassert resume['provenance']['target_revision'] == trev\n"
            s += "assert resume['provenance']['train_stream_sha256'] == TRAIN_SHA\n"
        if name == 'c-arb-math':
            s = '\n'.join(line for line in s.splitlines() if not line.startswith('#')) + '\n'
        if name == 'c-train-arb':
            s = s.replace('torch.cuda.reset_peak_memory_stats(0)', 'gpu_reset()')
            s = s.replace('arb_peak = torch.cuda.max_memory_allocated(0) / 1024**3', 'arb_gpu_peaks = gpu_peaks()\narb_peak = max(arb_gpu_peaks)')
            s = s.replace("WORK / 'arbitration'", "WORK / 'arbitration-%dm'" % stage)
            s = s.replace('    assert test[\'tokens\'] == tokens', "    assert test['tokens'] == tokens")
            s = s.replace("    os.replace(tmp, path)", "    os.replace(tmp, path)\n    fd = os.open(path.parent, os.O_RDONLY)\n    try: os.fsync(fd)\n    finally: os.close(fd)")
            s = s.replace("'tokens': tokens, 'optimizer_step': step,", "'tokens': tokens, 'optimizer_step': step,\n             'corpus_cursor': {'next_block':step, 'tokens':tokens, 'sha256':EVAL_MANIFEST['calibration']},\n             'reader_tokens': %d," % ({10:10000384,15:15000064}[stage]))
            s = s.replace("    for step in range(first, last):", """    assert not opt.state
    assert not set(params) & (set(model.parameters()) | set(inj.parameters()))
    assert all(not p.requires_grad for p in inj.parameters())
    assert all(p.requires_grad and p.dtype == torch.float32 for p in params)
    assert QUALIFICATION['passed_before_optimizer_step']
    for step in range(first, last):""")
            s = s.replace('clip_grad_norm_(params, 1.0)', 'clip_grad_norm_(params, 1.0, error_if_nonfinite=True)')
            s = s.replace('        opt.step()', '        opt.step()\n        assert set(opt.state) == set(params)')
        if name == 'c-evaluation':
            s = s.replace("'arb_time_s': arb_time,", "'reader_tokens': %d, 'arb_time_s': arb_time," % ({10:10000384,15:15000064}[stage]))
            s = s.replace("'host_rss_mib': rss_mb(),", "'host_rss_mib': rss_mb(),\n           'cache_bytes':sum((cache_dir / f).stat().st_size for f in ('rows.u8','addrs.u32')),")
            s = s.replace("'arb_peak_gib': arb_peak,", "'arb_peak_gib': arb_peak, 'arb_gpu_peaks_gib':arb_gpu_peaks,")
        cells.append(cell(name, s))
    save('arb-%dm' % stage, cells, 'qwengram-4b-%dm-linear750-evaluation-t4' % stage)


def controls():
    ref = json.loads((REF / 'controls/controls_500k.ipynb').read_text())
    main = json.loads((HERE / 'train-5m/train-5m.ipynb').read_text())
    cells = shared()
    cache = adapt(source(ref, 'c-compact'))
    cache += "\nassert CACHE_SHA == '209957565218351e1a8c44d7215462a7efa12cc1c2190c72bf48d5ef8c4bd4ae'\n"
    cache += "assert ADDR_SHA == 'a7eb85615b155b824a1d07cfc5a179879862cb061548fab17cd285b9fdb843c4'\n"
    cells.append(cell('c-compact', cache))
    cells.append(cell('c-load', adapt(source(BASE, 'c-load'))))
    cells.append(cell('c-qualification', preflight()))
    cells.append(cell('c-trainfn', source(main, 'c-trainfn')))
    run = adapt(source(ref, 'c-controls'))
    helpers = source(main, 'c-train-stage').split('RUN_META = ', 1)[0]
    helpers = helpers.replace("'reader-%d.safetensors' % seen", "'reader-%s-%d.safetensors' % (CURRENT, seen)")
    run = helpers + 'RUN_META = ' + run.split('RUN_META = ', 1)[1]
    run = run.replace("results = {'DISABLED': disabled}", "results = {'DISABLED': disabled}\ntraining = {}")
    run = run.replace("    seen = run['checkpoints'][-1]['tokens']", "    training[CURRENT] = run['checkpoints'][-1]\n    seen = run['checkpoints'][-1]['tokens']")
    run = run.replace("'budget_tokens': 500224,\n           'full_nll'", "'budget_tokens': 500224, 'training':training,\n           'full_nll'")
    cells.append(cell('c-controls', run))
    save('controls', cells, 'qwengram-4b-controls-500k-t4')


if __name__ == '__main__':
    for path, expected in PROTOCOL['reference_files'].items():
        assert sha((HERE.parent / path).read_bytes()) == expected, path
    assert source(BASE, 'c-hashing') == adapt(source(BASE, 'c-hashing'))
    assert source(BASE, 'c-ple') == adapt(source(BASE, 'c-ple'))
    benchmark()
    for stage in (5, 10, 15):
        reader(stage)
    for stage in (10, 15):
        arbitration(stage)
    controls()
    print('Built pinned 4B qualification, reader, arbitration, and control notebooks')
