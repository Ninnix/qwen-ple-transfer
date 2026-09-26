import json
from pathlib import Path

ROOT = Path(__file__).parent
REF = ROOT.parent / 'qwen35-08b/reader-scale-20m/kaggle_reader_20m_canonical.ipynb'
PIN = json.loads((ROOT / 'frozen.json').read_text())
ref = json.loads(REF.read_text())


def source(name):
    return ''.join(next(c['source'] for c in ref['cells'] if c.get('id') == name))


def cell(name, body):
    return {'cell_type': 'code', 'execution_count': None, 'id': name,
            'metadata': {}, 'outputs': [], 'source': body.splitlines(keepends=True)}


install = source('c-install')
install = install.replace('# Cell 1 — deps (P100-compatible torch BEFORE first torch import)', '# T4 dependencies')
install = install[:install.index("if _cap.startswith('6.'):")] + install[install.index('import torch\n'):]
install = install.replace('torch.zeros(1).cuda()  # fail fast if the build lacks sm_60 kernels',
                          "assert torch.cuda.get_device_capability(0) == (7, 5)")
assert install.count('%pip install') == 1

config = source('c-config').replace('Qwen/Qwen3.5-0.8B', PIN['target_model'])
config = config.replace('HIDDEN: int = 1024', 'HIDDEN: int = 2048')
config = config.replace('PLACEMENTS: tuple = ((2,), (8,), (2, 8))', 'PLACEMENTS: tuple = ((2, 8),)')
config = config.replace('for sites, R in [(1,1),(2,1),(1,4),(2,4)]:', 'for sites, R in [(2,1)]:')
assert 'N_LAYERS: int = 24' in config

reader = source('c-reader').replace('1024', '2048')
inject = source('c-inject').replace('1024', '2048')
load = source('c-load').replace('Qwen3.5-0.8B', 'Qwen3.5-2B').replace('hidden_size==1024', 'hidden_size==2048')
load = load.replace("tok=secret_value_0; assert tok, 'Attach HF_TOKEN in Settings -> Secrets'", "tok=secret_value_0")
load = load.replace('for dt in [torch.float16, torch.float32]:', 'for dt in [torch.float16]:')
load = load.replace("trev=HfApi(token=tok).model_info(C.TARGET_ID).sha", "trev='%s'" % PIN['target_revision'])
load = load.replace("assert model is not None and ACTIVE_DTYPE is not None", "assert model is not None and ACTIVE_DTYPE is not None")

tok = source('c-tok')
tok = tok.replace("tok=secret_value_0; assert tok, 'Attach HF_TOKEN in Settings -> Secrets'", "tok=secret_value_0")
tok = tok.replace('trev=api.model_info(C.TARGET_ID).sha; srev=api.model_info(C.SOURCE_ID).sha',
                  "trev='%s'; srev='%s'" % (PIN['target_revision'], PIN['source_revision']))
tok += "\nassert api.model_info(C.TARGET_ID).sha == trev\n"
tok += '''import hashlib
tp=hf_hub_download(C.TARGET_ID, 'tokenizer.json', revision=trev, token=tok)
sp=hf_hub_download(C.SOURCE_ID, 'tokenizer.json', revision=srev, token=tok)
assert hashlib.sha256(open(tp, 'rb').read()).hexdigest() == '%s'
assert hashlib.sha256(open(sp, 'rb').read()).hexdigest() == '%s'
assert json.load(open(tp))['model']['merges'] == json.load(open(sp))['model']['merges']
''' % (PIN['target_tokenizer_sha256'], PIN['source_tokenizer_sha256'])

bench = '''import json, time, torch, torch.nn.functional as F
from pathlib import Path

assert C.HIDDEN == 2048 and C.N_LAYERS == 24 and C.VOCAB == 248320
assert torch.cuda.device_count() == 2
assert tuple(torch.cuda.get_device_capability(i) for i in range(2)) == ((7, 5), (7, 5))
assert not any(p.requires_grad for p in model.parameters())

torch.manual_seed(1234)
ids = tokenizer('The quick brown fox jumps over the lazy dog. ' * 150,
                return_tensors='pt').input_ids[:, :512].cuda()
assert ids.shape == (1, 512)
memory = torch.randn(1, 512, C.MEM_DIM, device='cuda') * 0.0086
inj = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
assert inj.idx == (2, 8)

with torch.inference_mode():
    inj.set_memory(None)
    stock = model(input_ids=ids, use_cache=False).logits
    inj.set_memory(memory)
    for r in inj.readers.values(): r.gamma.zero_()
    disabled = model(input_ids=ids, use_cache=False).logits
    dmax = (stock.float() - disabled.float()).abs().max().item()
assert dmax <= 0.002, dmax
with torch.no_grad():
    for r in inj.readers.values(): r.gamma.fill_(C.GAMMA_INIT)

torch.cuda.reset_peak_memory_stats(0)
with torch.inference_mode():
    for _ in range(2): model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    for _ in range(4): model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize(0)
    forward_s = time.perf_counter() - start
forward_peak = torch.cuda.max_memory_allocated(0)

params = list(inj.parameters())
assert len(params) == 8 and all(p.dtype == torch.float32 for p in params)
opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=C.WD)
assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in params}
torch.cuda.reset_peak_memory_stats(0)
start = time.perf_counter()
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, use_cache=False).logits
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
torch.cuda.synchronize(0)
train_s = time.perf_counter() - start

report = {'target_revision': trev, 'dtype': str(ACTIVE_DTYPE),
          'placement_idx': [2, 8], 'identity_max_abs_logit': dmax,
          'forward_tok_s': 4 * 512 / forward_s,
          'training_tok_s': 3 * 512 / train_s,
          'forward_peak_gib': forward_peak / 1024**3,
          'training_peak_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
          'gpu1_allocated_gib': torch.cuda.memory_allocated(1) / 1024**3,
          'host_rss_mib': rss_mb(), 'reader_params': sum(p.numel() for p in params),
          'loss': float(loss.detach())}
Path('/kaggle/working/benchmark-2b.json').write_text(json.dumps(report, indent=2))
print('BENCHMARK_2B', json.dumps(report), flush=True)
inj.close()
'''

names = ('c-env',)
env = source('c-env').replace("assert secret_value_0 and secret_value_0.startswith('hf_'), 'set HF_TOKEN or attach Kaggle HF_TOKEN secret'", 'pass')
cells = [cell('c-env', env), cell('c-install', install), cell('c-config', config)]
for name in ('c-hashing', 'c-reader', 'c-inject', 'c-ple'):
    cells.append(cell(name, {'c-reader': reader, 'c-inject': inject}.get(name, source(name))))
cells += [cell('c-tok', tok), cell('c-load', load), cell('c-benchmark', bench)]
out = {'cells': cells, 'metadata': ref['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
(ROOT / 'benchmark.ipynb').write_text(json.dumps(out, indent=1))
