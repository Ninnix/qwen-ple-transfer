import json
from pathlib import Path

ROOT = Path(__file__).parent
REF = json.loads((ROOT / 'train-5m/train_5m.ipynb').read_text())
PIN = json.loads((ROOT / 'frozen.json').read_text())
OUT = ROOT / 'arb-eval'
OUT.mkdir(exist_ok=True)


def source(name):
    return ''.join(next(c['source'] for c in REF['cells'] if c.get('id') == name))


def cell(name, body):
    return {'cell_type': 'code', 'execution_count': None, 'id': name,
            'metadata': {}, 'outputs': [], 'source': body.splitlines(keepends=True)}


names = ('c-env', 'c-install', 'c-config', 'c-hashing', 'c-reader',
         'c-inject', 'c-ple', 'c-tok', 'c-load', 'c-data')
cells = [cell(name, source(name)) for name in names]
valprep = source('c-valprep')
valprep += '''
for name in ('code__evaluate.py', 'code__bootstrap.py'):
    src = STAGE / name
    assert hashlib.sha256(src.read_bytes()).hexdigest() == _man[name]['sha256']
    shutil.copyfile(src, WORK / name.split('__', 1)[1])
eval_hits = list(Path('/kaggle/input').rglob('evaluation-streams.json'))
assert len(eval_hits) == 1
prior_eval = json.loads(eval_hits[0].read_text())
for key in ('full', 'fast', 'domains', 'calibration', 'hellaswag', 'lambada'):
    assert EVAL_MANIFEST[key] == prior_eval[key], key
'''
cells.append(cell('c-valprep', valprep))

cache = '''import hashlib, json, shutil, time
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

assert len(find_ple_manifests()) == 11
master = MountPLE()
cache_dir = WORK / 'arb-cache'
cache_dir.mkdir(parents=True, exist_ok=True)
for name, srcname in (('compact.json', 'compact__compact.json'),
                      ('addrs.u32', 'compact__addrs.u32'),
                      ('rows.u8', 'compact__rows.u8')):
    src = STAGE / srcname
    assert src.stat().st_size == _man[srcname]['size']
    assert sha(src) == _man[srcname]['sha256']
    dst = cache_dir / name
    if not dst.exists() or sha(dst) != _man[srcname]['sha256']:
        shutil.copyfile(src, dst)
    assert sha(dst) == _man[srcname]['sha256']
store = CompactPLE(cache_dir)
assert store.ple_revision == '%s'
g = torch.Generator().manual_seed(0)
sample = store.addrs[torch.randint(0, len(store.addrs), (2048,), generator=g)].reshape(128, 16)
maxdiff = (store.lookup(sample) - master.lookup(sample)).abs().max().item()
assert maxdiff == 0.0
cache_meta = store.meta
print('arb compact cache verified', cache_meta['rows_sha256'],
      cache_meta['addrs_sha256'], maxdiff, flush=True)
''' % PIN['source_revision']
cells.append(cell('c-cache-arb', cache))

reader = '''from safetensors.torch import load_file
summary_hits = list(Path('/kaggle/input').rglob('reader-15m-summary.json'))
assert len(summary_hits) == 1
reader_summary = json.loads(summary_hits[0].read_text())
reader_path = summary_hits[0].parent / 'reader-15000064.safetensors'
assert sha(reader_path) == reader_summary['reader_sha256']
resume_path = summary_hits[0].parent / 'reader-real-15m-15000064.pt'
assert sha(resume_path) == reader_summary['checkpoint_sha256']
resume = torch.load(resume_path, map_location='cpu', weights_only=False)
assert resume['seen'] == 15000064 and resume['cfg'] == ((2, 8), 1)
assert resume['provenance']['train_stream_sha256'] == json.loads((summary_hits[0].parent / 'reader-train-15m.json').read_text())['sha256']
inj = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
weights = load_file(str(reader_path), device='cpu')
assert set(weights) == set(inj.state_dict())
assert all(torch.equal(weights[k], resume['reader'][k]) for k in weights)
inj.load_state_dict({k: v.to('cuda') for k, v in weights.items()})
inj.requires_grad_(False)
inj.set_memory(None)
assert not any(p.requires_grad for p in model.parameters())
GAMMAS = {site: float(inj.readers[site].gamma.detach()) for site in ('2', '8')}
UPDATE_NORMS = {}
for key, value in resume['reader'].items():
    if key.endswith(('keys.0.weight', 'value.weight', 'beta')):
        UPDATE_NORMS[key] = float((value.float() - resume['init_state'][key].float()).norm())
print('15M frozen reader loaded', GAMMAS, flush=True)
'''
cells.append(cell('c-reader-load', reader))

arb = (ROOT.parent / 'qwen35-08b/lightning/arb_frozen.py').read_text().replace('hidden=1024', 'hidden=2048').replace('(~1K)', '(~2K)')
cells.append(cell('c-arb-math', arb))

train = '''import math, os, random, torch.nn.functional as F
from array import array

cal_path = STAGE / 'calib__calib-750k-blocks.uint32le'
assert sha(cal_path) == EVAL_MANIFEST['calibration']
a = array('I'); a.frombytes(cal_path.read_bytes())
CAL = torch.tensor(a, dtype=torch.long).view(-1, 512)
assert CAL.shape == (1464, 512)
torch.manual_seed(1234); random.seed(1234)
arb = MemoryArbitration(hidden=2048).to('cuda')
assert abs(float(arb.alpha2().detach()) - 1.25) < 1e-7
assert abs(float(arb.alpha8(torch.zeros(1, 1, 2048, device='cuda')).detach()) - 0.125) < 1e-7
assert arb.w.numel() == 2048 and torch.count_nonzero(arb.w) == 0
params = list(arb.parameters())
assert len(params) == 3 and sum(p.numel() for p in params) == 2050
hooks = ArbHooks(model, inj, arb, decoder_layers)
assert hooks.fixed_alpha2 is None
work = WORK / 'arbitration'
work.mkdir(exist_ok=True)
reader_sha = sha(reader_path)
ple_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in master.part_paths.values()}
model_versions = {n: p._version for n, p in model.named_parameters()}
reader_versions = {n: p._version for n, p in inj.named_parameters()}
torch.cuda.reset_peak_memory_stats(0)
start_time = time.perf_counter()

def lr_at(step, total):
    if step < 32: return 1e-3 * (step + 1) / 32
    x = min((step - 32) / max(1, total - 32), 1.0)
    return 5e-4 * (1 + math.cos(math.pi * x))

def save_arb(tokens, step, opt, schedule):
    path = work / ('linear-%d.pt' % tokens)
    tmp = path.with_suffix('.pt.tmp')
    state = {'alpha2_raw': arb.alpha2_raw.detach().cpu(),
             'w': arb.w.detach().cpu(), 'b': arb.b.detach().cpu(),
             'optimizer': opt.state_dict(), 'scheduler': schedule,
             'tokens': tokens, 'optimizer_step': step,
             'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
             'rng_python': random.getstate(),
             'reader_sha256': reader_sha,
             'calibration_sha256': EVAL_MANIFEST['calibration'],
             'compact_cache_sha256': cache_meta['rows_sha256'],
             'address_mapping_sha256': cache_meta['addrs_sha256'],
             'target_revision': '__TARGET_REV__', 'tokenizer_sha256': '__TOKENIZER_SHA__',
             'ple_revision': '__PLE_REV__', 'injection_idx': [2, 8],
             'loss': 'causal CE + 2*(mean(alpha_late)-0.125)^2'}
    with tmp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    test = torch.load(tmp, map_location='cpu', weights_only=False)
    assert test['tokens'] == tokens and test['reader_sha256'] == reader_sha
    assert test['optimizer_step'] == step and len(test['optimizer']['state']) == 3
    assert all(torch.equal(test[k], state[k]) for k in ('alpha2_raw', 'w', 'b'))
    os.replace(tmp, path)
    print('ARB_CHECKPOINT', tokens, sha(path), flush=True)
    return path

for first, last in ((0, 978), (978, 1464)):
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    assert {id(p) for group in opt.param_groups for p in group['params']} == {id(p) for p in params}
    for step in range(first, last):
        lr = lr_at(step - first, last - first)
        for group in opt.param_groups: group['lr'] = lr
        ids = CAL[step].unsqueeze(0)
        hooks.set_memory(store.lookup(addresses(ids)).to('cuda'))
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=ids.to('cuda'), use_cache=False).logits
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             ids.to('cuda')[:, 1:].reshape(-1))
        a8 = hooks.cur_alpha8
        loss = ce + 2.0 * (a8.float().mean() - 0.125).square()
        assert torch.isfinite(loss)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters())
        assert not any(p.grad is not None for p in inj.parameters())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print('arb step', step + 1, 'CE', float(ce.detach()),
                  'early', float(arb.alpha2().detach()),
                  'late', float(a8.detach().mean()), flush=True)
    save_arb(last * 512, last, opt, {'leg_start': first, 'leg_end': last,
             'warm_steps': 32, 'lr_peak': 1e-3, 'last_lr': lr})
    assert ple_stats == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in master.part_paths.values()}
    assert model_versions == {n: p._version for n, p in model.named_parameters()}
    assert reader_versions == {n: p._version for n, p in inj.named_parameters()}
arb_time = time.perf_counter() - start_time
arb_peak = torch.cuda.max_memory_allocated(0) / 1024**3
arb.requires_grad_(False)
hooks.close(); inj.set_memory(None)
assert abs(float(arb.alpha2().detach()) - 1.25) > 1e-6
print('LINEAR750_COMPLETE', arb_time, arb_peak, flush=True)
'''
train = train.replace('__TARGET_REV__', PIN['target_revision']).replace('__TOKENIZER_SHA__', PIN['target_tokenizer_sha256']).replace('__PLE_REV__', PIN['source_revision'])
cells.append(cell('c-train-arb', train))

evaluation = '''import evaluate, bootstrap
from random import Random

val = val_full
domains = {d: frozen_data.load_blocks(STAGE / ('frozen-v1__tokens-%s.uint32le' % d), EVAL_MANIFEST['domains'][d])
           for d in ('general', 'code', 'math', 'scientific', 'multilingual')}

def score_static(label, zero_gamma):
    old = {k: p.detach().clone() for k, p in inj.named_parameters() if k.endswith('gamma')}
    if zero_gamma:
        with torch.no_grad():
            for k, p in inj.named_parameters():
                if k in old: p.zero_()
    result = evaluate.eval_all_static(model, inj, store, addresses, tokenizer,
                                      val, domains, hellaswag, lambada,
                                      time.perf_counter(), 1e12)
    with torch.no_grad():
        for k, p in inj.named_parameters():
            if k in old: p.copy_(old[k])
    assert len(result['val_blocks']) == 1024
    assert len(result['hs_correct']) == len(result['lam_correct']) == 1000
    (work / ('eval-%s.json' % label)).write_text(json.dumps(result))
    print('EVAL', label, result['val_nll'], result['domain_mean'], flush=True)
    return result

stock = score_static('stock', True)
raw = score_static('raw-15m', False)
inj.set_memory(None)
hooks = ArbHooks(model, inj, arb, decoder_layers)
final = evaluate.eval_all(model, hooks, store, addresses, tokenizer,
                          val, domains, hellaswag, lambada,
                          time.perf_counter(), 1e12)
hooks.close()
assert len(final['val_blocks']) == 1024
assert len(final['hs_correct']) == len(final['lam_correct']) == 1000
assert all(len(final['dom_blocks'][d]) == 64 for d in domains)
(work / 'eval-linear750.json').write_text(json.dumps(final))

def paired(a, b, name):
    out = bootstrap.contrast(name, a, b)
    rng = Random(1234)
    dd = {d: [x['sum']/x['n'] - y['sum']/y['n']
              for x, y in zip(a['dom_blocks'][d], b['dom_blocks'][d])]
          for d in sorted(domains)}
    reps = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v)/len(v)
                      for v in dd.values())/len(dd) for _ in range(10000))
    out['domain_mean_nll'] = {'mean': sum(sum(v)/len(v) for v in dd.values())/len(dd),
                              'lo': reps[250], 'hi': reps[9749]}
    out['paired_deltas'] = {'val_nll': [x['sum']/x['n'] - y['sum']/y['n']
                                       for x, y in zip(a['val_blocks'], b['val_blocks'])],
                            'domains': dd,
                            'hellaswag_acc': [x-y for x,y in zip(a['hs_correct'], b['hs_correct'])],
                            'lambada_acc': [x-y for x,y in zip(a['lam_correct'], b['lam_correct'])],
                            'lambada_nll': [x-y for x,y in zip(a['lam_nlls'], b['lam_nlls'])]}
    return out

contrasts = {'raw_minus_stock': paired(raw, stock, 'raw - stock'),
             'linear_minus_stock': paired(final, stock, 'linear750 - stock'),
             'linear_minus_raw': paired(final, raw, 'linear750 - raw')}
(work / 'bootstrap.json').write_text(json.dumps(contrasts))

def update_norms():
    layers = decoder_layers(model)
    captured = {}
    handles = []
    for idx in (2, 8):
        def before(mod, args, kwargs, site=idx):
            captured[site] = (args[0] if args else kwargs['hidden_states']).detach().clone()
        handles.append(layers[idx].register_forward_pre_hook(before, with_kwargs=True, prepend=True))
    active = ArbHooks(model, inj, arb, decoder_layers)
    sums = {site: {'reader': 0.0, 'effective': 0.0, 'relative': 0.0} for site in (2, 8)}
    count = 0
    with torch.inference_mode():
        for block in val[:64]:
            ids = block.unsqueeze(0)
            memory = store.lookup(addresses(ids)).to('cuda')
            active.set_memory(memory)
            model(input_ids=ids.to('cuda'), use_cache=False)
            for site in (2, 8):
                h = captured[site]
                delta = inj.readers[str(site)](h, memory).float() - h.float()
                alpha = arb.alpha2() if site == 2 else arb.alpha8(h)
                effective = delta * alpha.float().unsqueeze(-1) if site == 8 else delta * alpha.float()
                sums[site]['reader'] += float(delta.norm(dim=-1).mean())
                sums[site]['effective'] += float(effective.norm(dim=-1).mean())
                sums[site]['relative'] += float((effective.norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-6)).mean())
            count += 1
    active.close()
    for handle in handles: handle.remove()
    return {str(site): {key: value/count for key, value in vals.items()}
            for site, vals in sums.items()}

RESIDUAL_NORMS = update_norms()
summary = {'stock': {k: stock[k] for k in ('val_nll', 'domain_mean', 'domains', 'hs_acc', 'lambada_acc', 'lambada_nll')},
           'raw': {k: raw[k] for k in ('val_nll', 'domain_mean', 'domains', 'hs_acc', 'lambada_acc', 'lambada_nll')},
           'linear750': {k: final[k] for k in ('val_nll', 'domain_mean', 'domains', 'hs_acc', 'lambada_acc', 'lambada_nll', 'alpha8')},
           'early_alpha': float(arb.alpha2().detach()),
           'gammas': GAMMAS,
           'reader_parameter_update_norms': UPDATE_NORMS,
           'reader_residual_update_norms': RESIDUAL_NORMS,
           'effective_early': GAMMAS['2'] * float(arb.alpha2().detach()),
           'effective_late_mean': GAMMAS['8'] * final['alpha8']['full-val']['mean'],
           'arb_time_s': arb_time, 'arb_peak_gib': arb_peak,
           'arb_training_tok_s': 749568 / arb_time,
           'host_rss_mib': rss_mb(),
           'arb_checkpoint_sha256': sha(work / 'linear-749568.pt'),
           'reader_sha256': reader_sha,
           'cache_sha256': cache_meta['rows_sha256'],
           'mapping_sha256': cache_meta['addrs_sha256'],
           'eval_streams_sha256': sha(WORK / 'evaluation-streams.json')}
(work / 'summary.json').write_text(json.dumps(summary, indent=2))
print('QWENGRAM_2B_EVAL_COMPLETE', json.dumps(summary), flush=True)
'''
cells.append(cell('c-evaluation', evaluation))

nb = {'cells': cells, 'metadata': REF['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
(OUT / 'arb_eval.ipynb').write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / 'train-5m/kernel-metadata.json').read_text())
meta.update(code_file='arb_eval.ipynb', id='ninnix/qwengram-2b-linear750-evaluation-t4',
            title='Qwengram 2B linear750 evaluation T4',
            dataset_sources=meta['dataset_sources'] + ['ninnix/qwengram-2b-checkpoints'])
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
