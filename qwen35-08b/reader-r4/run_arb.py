import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/kaggle/working/reader-arb-r4')
ROOT.mkdir(parents=True, exist_ok=True)

hf = os.environ.get('HF_TOKEN')
if not hf:
    try:
        from kaggle_secrets import UserSecretsClient
        hf = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        pass
assert hf and hf.startswith('hf_')
os.environ['HF_TOKEN'] = hf
os.environ['LIGHTNING_PERSISTENT_DIR'] = str(ROOT)

cap = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True).splitlines()[0].strip()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.17.0', 'datasets', 'safetensors', 'huggingface_hub', 'accelerate'], check=True)
if cap.startswith('6.'):
    pytorch = ['--index-url', 'https://download.pytorch.org/whl/cu118']
    for package in ('torch==2.5.1+cu118', 'torchvision==0.20.1+cu118', 'torchaudio==2.5.1+cu118'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pytorch, package], check=True)
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
assert torch.cuda.is_available()
torch.zeros(1, device='cuda')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def find(name):
    hits = list(Path('/kaggle/input').rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]

stage = find('transfer-shas.json')
manifest = json.loads(stage.read_text())
for name, meta in manifest.items():
    src = stage.parent / name
    assert src.stat().st_size == meta['size'] and sha(src) == meta['sha256'], name
    group, base = name.split('__', 1)
    dst = ROOT / ('code' if group == 'code' else group) / base
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == meta['sha256'], name
print('frozen evaluation and calibration stage verified', len(manifest), flush=True)

r4_manifest = find('reader-r4-shas.json')
r4_files = json.loads(r4_manifest.read_text())
assert r4_files['source_kernel'] == 'ninnix/qwen-ple-real-r4-reader-warm-start-early-1m-t4/3'
for name, meta in r4_files['files'].items():
    src = r4_manifest.parent / name
    assert src.stat().st_size == meta['size'] and sha(src) == meta['sha256'], name
    dst = ROOT / 'readers' / name
    dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == meta['sha256'], name
canonical = json.loads((ROOT / 'readers' / 'canonical-15m.json').read_text())
assert canonical['reader_tokens'] == 15000064 and canonical['arbiter_calibration_tokens'] == 749568
assert canonical['reader_weights_sha256'] == 'e4a760163ec07568178ab48aa533235a9af878183caf120d4e29ce1c8ce4b9dc'
print('R4 milestone and 15M parent SHA verified', flush=True)

prior_manifest = find('arb-15m-shas.json')
prior = json.loads(prior_manifest.read_text())
assert prior['source_kernel'] == 'ninnix/qwen-ple-reader-scale-15m-arbitration-p100/1'
for name in ('arb-reader15m-arb750-749568.pt', 'eval-reader15m-arb750.json',
             'alpha8-reader15m-arb750.pt', 'checkpoints.json'):
    src = prior_manifest.parent / name
    meta = prior['files'][name]
    assert src.stat().st_size == meta['size'] and sha(src) == meta['sha256'], name
    dst = ROOT / 'prior15' / name
    dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == meta['sha256'], name
old = json.loads((ROOT / 'prior15' / 'checkpoints.json').read_text())
assert old['reader_shas']['15m'] == canonical['reader_weights_sha256']
assert old['arbitration_shas']['reader15m-arb750'] == canonical['arbiter_checkpoint_sha256']
baseline = json.loads((ROOT / 'prior15' / 'eval-reader15m-arb750.json').read_text())
assert abs(baseline['val_nll'] - canonical['full_val_nll']) < 1e-12
assert abs(baseline['domain_mean'] - canonical['five_domain_mean_nll']) < 1e-12
assert abs(baseline['lambada_nll'] - canonical['lambada_nll']) < 1e-12
assert len(baseline['val_blocks']) == 1024 and len(baseline['hs_correct']) == 1000

code = ROOT / 'code'
src = (code / 'evaluate.py').read_text()
needle = '    stats = {k: alpha_stats(torch.cat(vs)) for k, vs in collect.items() if vs}\n'
assert src.count(needle) == 1
src = src.replace(needle, "    raw = __import__('os').environ.get('PLE_ALPHA8_RAW')\n    if raw: torch.save({k: [v.cpu() for v in vs] for k, vs in collect.items()}, raw)\n" + needle)
(code / 'evaluate.py').write_text(src)
sys.path.insert(0, str(code))
import arb_frozen
import bootstrap
import config
import evaluate
import frozen_data
import studio_flow
import train as TR

tok, model, inj, versions, ngram_fn, FZ = studio_flow.load_stack(hf)
inj.close()
inj = FZ.ReaderInjection(model, [2,8], 2560, 1024, 4, 1e-3).to('cuda')
store = FZ.CompactPLE(str(ROOT / 'compact'))
val = frozen_data.load_blocks(ROOT / 'frozen-v1' / 'tokens-full.uint32le', config.REVS['val_full_sha256'])
domains = {d:frozen_data.load_blocks(ROOT / 'frozen-v1' / ('tokens-%s.uint32le' % d), config.REVS['domains'][d]) for d in frozen_data.DOMAINS}
hs, _ = frozen_data.hellaswag_rows(1000, hf)
lam, _ = frozen_data.lambada_rows(1000, hf)
assert len(val) == 1024 and len(hs) == len(lam) == 1000 and all(len(v) == 64 for v in domains.values())
full, full_man = TR.ordered_full(ROOT / 'calib')
for tag, blocks in (('500', 978), ('750', 1464)):
    assert hashlib.sha256(full[:blocks].numpy().astype('<u4').tobytes()).hexdigest() == full_man['milestones'][tag]['sha256']

reader_path = ROOT / 'readers' / 'real-r4-early.reader.safetensors'
reader_sha = r4_files['files'][reader_path.name]['sha256']
assert sha(reader_path) == reader_sha
reader = load_file(str(reader_path), device='cpu')
assert set(reader) == set(inj.state_dict())
inj.load_state_dict({k:v.to('cuda') for k,v in reader.items()})
inj.requires_grad_(False)
inj.set_memory(None)
gamma = {n:float(p.detach()) for n,p in inj.named_parameters() if n.endswith('gamma')}
assert not any(p.requires_grad for p in model.parameters())
print('R4 reader loaded', reader_sha, gamma, flush=True)

def score(label, arb):
    hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
    hooks.fixed_alpha2 = config.CANON['alpha2_frozen']
    os.environ['PLE_ALPHA8_RAW'] = str(ROOT / ('alpha8-%s.pt' % label))
    try:
        with torch.inference_mode():
            rep = evaluate.eval_all(model, hooks, store, ngram_fn, tok, val, domains, hs, lam,
                                    time.perf_counter(), studio_flow.HARD_S)
    finally:
        hooks.close()
    assert len(rep['val_blocks']) == 1024
    assert len(rep['hs_correct']) == len(rep['lam_correct']) == len(rep['lam_nlls']) == 1000
    assert all(len(v) == 64 for v in rep['dom_blocks'].values())
    assert set(rep['alpha8']) == {'full-val', 'hellaswag', 'lambada', *domains}
    assert Path(os.environ['PLE_ALPHA8_RAW']).exists()
    (ROOT / ('eval-%s.json' % label)).write_text(json.dumps(rep))
    print(label, 'full-val', rep['val_nll'], 'domain mean', rep['domain_mean'],
          'LAMBADA', rep['lambada_nll'], flush=True)
    return rep

def checkpoint(label, arb, tokens):
    path = ROOT / ('arb-%s-%d.pt' % (label, tokens))
    state = {'w':arb.w.detach().cpu(), 'b':arb.b.detach().cpu(),
             'reader_sha256':reader_sha, 'calibration_tokens':tokens,
             'alpha2_frozen':config.CANON['alpha2_frozen'],
             'optimizer':'fresh AdamW per leg, lr_peak=1e-3, wd=0, warmup=32, cosine',
             'loss':'cross-entropy + 2*(mean(alpha8)-0.125)^2'}
    digest, reopened = TR.atomic_torch_save(state, path, ('w', 'b', 'reader_sha256'))
    assert digest == sha(path) and reopened['calibration_tokens'] == tokens
    assert torch.equal(reopened['w'], state['w']) and torch.equal(reopened['b'], state['b'])
    print('arb checkpoint SHA/reopen', label, tokens, digest, flush=True)
    return digest

torch.manual_seed(1234)
arb = arb_frozen.MemoryArbitration().to('cuda')
arb.alpha2_raw.requires_grad_(False)
assert torch.count_nonzero(arb.w) == 0
assert abs(float(arb.alpha8(torch.zeros(1, 1, 1024, device='cuda')).detach()) - 0.125) < 1e-7
hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
hooks.fixed_alpha2 = config.CANON['alpha2_frozen']
checkpoints = {}
for tag, start, end in ((500, 0, 978), (750, 978, 1464)):
    params = [p for p in arb.parameters() if p.requires_grad]
    assert len(params) == 2 and sum(p.numel() for p in params) == 1025
    opt = torch.optim.AdamW(params, lr=config.ARB['lr_peak'], weight_decay=config.ARB['wd'])
    for block in range(start, end):
        for group in opt.param_groups:
            group['lr'] = TR.lr_at(block - start, end - start)
        ids = full[block].unsqueeze(0)
        hooks.set_memory(store.lookup(ngram_fn(ids)).to('cuda'))
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=ids.to('cuda'), use_cache=False).logits
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids.to('cuda')[:, 1:].reshape(-1))
        a8 = hooks.cur_alpha8
        loss = ce + config.ARB['lambda_mean_reg'] * (a8.float().mean() - config.ARB['alpha8_target']) ** 2
        assert torch.isfinite(loss)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters())
        assert not any(p.grad is not None for p in inj.parameters())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (block + 1) % 100 == 0:
            print('R4 arb tokens', (block + 1)*512, 'ce', float(ce.detach()),
                  'mean alpha8', float(a8.detach().mean()), flush=True)
        assert not [n for n,p in model.named_parameters() if p._version != versions[n]]
        del logits, ce, a8, loss
    label = 'reader-r4-arb%d' % tag
    checkpoints[label] = checkpoint(label, arb, end*512)
    del opt
hooks.close()
arb.requires_grad_(False)
result = score('reader-r4-arb750', arb)

contrast = bootstrap.contrast('reader-r4-arb750-vs-reader15m-linear750', result, baseline)
from random import Random
rng = Random(1234)
dd = {d:[x['sum']/x['n'] - y['sum']/y['n'] for x,y in zip(result['dom_blocks'][d], baseline['dom_blocks'][d])] for d in sorted(domains)}
reps = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v)/len(v) for v in dd.values())/len(dd) for _ in range(10000))
contrast['domain_mean_nll'] = {'mean':sum(sum(v)/len(v) for v in dd.values())/len(dd),
                               'lo':reps[250], 'hi':reps[9749],
                               'p':min(1.0, 2*min(sum(x>=0 for x in reps), sum(x<=0 for x in reps))/len(reps))}
contrast['paired_deltas'] = {'val_nll':[x['sum']/x['n']-y['sum']/y['n'] for x,y in zip(result['val_blocks'],baseline['val_blocks'])],
                             'domains':dd,
                             'hellaswag_acc':[x-y for x,y in zip(result['hs_correct'],baseline['hs_correct'])],
                             'lambada_acc':[x-y for x,y in zip(result['lam_correct'],baseline['lam_correct'])],
                             'lambada_nll':[x-y for x,y in zip(result['lam_nlls'],baseline['lam_nlls'])]}
print(bootstrap.summarize(contrast), flush=True)

import numpy as np
def item_stats(rows, dataset):
    if dataset == 'hellaswag':
        assert len(rows) == 4000
        rows = [torch.cat(rows[i:i+4]) for i in range(0, len(rows), 4)]
    out = np.empty((len(rows), 5), dtype=np.float64)
    for i, row in enumerate(rows):
        x = row.numpy().astype(np.float64)
        out[i] = x.mean(), x.std(), *np.quantile(x, (0.05, 0.5, 0.95))
    return out

raw_r4 = torch.load(ROOT / 'alpha8-reader-r4-arb750.pt', map_location='cpu', weights_only=False)
raw15 = torch.load(ROOT / 'prior15' / 'alpha8-reader15m-arb750.pt', map_location='cpu', weights_only=False)
assert set(raw_r4) == set(raw15) == {'full-val', 'hellaswag', 'lambada', *domains}
alpha_delta = {}
for dataset in raw_r4:
    a = item_stats(raw_r4[dataset], dataset)
    b = item_stats(raw15[dataset], dataset)
    assert a.shape == b.shape
    delta = a - b
    n = len(delta)
    rng = np.random.default_rng(1234)
    picks = rng.integers(n, size=(10000, n), dtype=np.int32)
    alpha_delta[dataset] = {}
    for j, key in enumerate(('mean', 'std', 'p05', 'p50', 'p95')):
        reps = np.sort(delta[picks, j].mean(axis=1))
        p = min(1.0, 2*min(np.count_nonzero(reps >= 0), np.count_nonzero(reps <= 0))/len(reps))
        alpha_delta[dataset][key] = {'global_delta':result['alpha8'][dataset][key]-baseline['alpha8'][dataset][key],
                                     'paired_item_mean_delta':float(delta[:,j].mean()),
                                     'ci95':[float(reps[250]),float(reps[9749])],
                                     'p':float(p),'n':n,'paired_item_deltas':delta[:,j].tolist()}
(ROOT / 'alpha8-paired.json').write_text(json.dumps(alpha_delta, indent=2))
(ROOT / 'summary.json').write_text(json.dumps({'reader15m-linear750':baseline,'reader-r4-arb750':result}, indent=2))
(ROOT / 'contrast.json').write_text(json.dumps(contrast, indent=2))
(ROOT / 'checkpoints.json').write_text(json.dumps({
    'reader_source':r4_files['source_kernel'], 'reader_sha256':reader_sha,
    'reader_parent_sha256':canonical['reader_weights_sha256'],
    'arbitration_shas':checkpoints,
    'canonical15_arbiter_sha256':canonical['arbiter_checkpoint_sha256'],
    'baseline_eval_sha256':prior['files']['eval-reader15m-arb750.json']['sha256'],
    'gamma_r4':gamma,'alpha2_frozen':config.CANON['alpha2_frozen'],
    'calibration_shas':{k:full_man['milestones'][k]['sha256'] for k in ('500','750')},
    'frozen_eval_shas':config.REVS}, indent=2))
print('R4 EARLY ARBITRATION COMPLETE', flush=True)
