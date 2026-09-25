import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/kaggle/working/ple-residual')
ROOT.mkdir(parents=True, exist_ok=True)

hf = os.environ.get('HF_TOKEN')
if not hf:
    try:
        from kaggle_secrets import UserSecretsClient
        hf = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        pass
assert hf and hf.startswith('hf_'), 'HF_TOKEN unavailable'
os.environ['HF_TOKEN'] = hf
os.environ['LIGHTNING_PERSISTENT_DIR'] = str(ROOT)

cap = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True).splitlines()[0].strip()
print('compute cap', cap, flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.17.0', 'datasets', 'safetensors', 'huggingface_hub', 'accelerate'], check=True)
if cap.startswith('6.'):
    pytorch = ['--index-url', 'https://download.pytorch.org/whl/cu118']
    for package in ('torch==2.5.1+cu118', 'torchvision==0.20.1+cu118', 'torchaudio==2.5.1+cu118'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pytorch, package], check=True)
import torch
import torch.nn.functional as F
from torch import nn
assert torch.cuda.is_available()
torch.zeros(1, device='cuda')

class ResidualMLPArbitration(nn.Module):
    def __init__(self, w, b, width=32):
        super().__init__()
        assert w.numel() == 1024 and b.numel() == 1
        self.register_buffer('w', w.detach().float().clone())
        self.register_buffer('b', b.detach().float().clone())
        self.register_buffer('alpha2_raw', torch.tensor(0.0))
        self.fc1 = nn.Linear(1024, width)
        self.fc2 = nn.Linear(width, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def alpha8(self, h8):
        h = h8.detach().float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        linear = (h * self.w).sum(-1) + self.b
        delta = self.fc2(F.silu(self.fc1(h))).squeeze(-1)
        return 0.5 * torch.sigmoid(linear + delta)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def find_manifest(name):
    hits = list(Path('/kaggle/input').rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]

stage = find_manifest('transfer-shas.json')
manifest = json.loads(stage.read_text())
for name, meta in manifest.items():
    src = stage.parent / name
    assert src.stat().st_size == meta['size'] and sha(src) == meta['sha256'], name
    group, base = name.split('__', 1)
    dst = ROOT / ('code' if group == 'code' else group) / base
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == meta['sha256'], name
print('frozen stack verified:', len(manifest), 'files', flush=True)

ref_manifest = find_manifest('baseline-shas.json')
ref = json.loads(ref_manifest.read_text())
assert ref['source_kernel'] == 'ninnix/qwen-ple-idx8-arbiter-continuation-p100/4'
for name, meta in ref['files'].items():
    src = ref_manifest.parent / name
    assert src.stat().st_size == meta['size'] and sha(src) == meta['sha256'], name
    dst = ROOT / 'linear-ref' / name
    dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == meta['sha256'], name
print('linear 750K/1M references SHA verified', flush=True)

code = ROOT / 'code'
src = (code / 'evaluate.py').read_text()
needle = '    stats = {k: alpha_stats(torch.cat(vs)) for k, vs in collect.items() if vs}\n'
assert src.count(needle) == 1
src = src.replace(needle, "    raw = __import__('os').environ.get('PLE_ALPHA8_RAW')\n    if raw: torch.save({k: [v.cpu() for v in vs] for k, vs in collect.items()}, raw)\n" + needle)
(code / 'evaluate.py').write_text(src)
sys.path[:0] = [str(code), str(Path(__file__).parent)]
import arb_frozen
import bootstrap
import config
import evaluate
import frozen_data
import studio_flow
import train as TR

baseline = json.loads((ROOT / 'linear-ref' / 'linear-summary.json').read_text())
assert baseline['749568']['total_tokens'] == 749568
assert baseline['1001472']['total_tokens'] == 1001472
linear750 = ROOT / 'milestones' / 'arbitration-cont-750k.pt'
assert sha(linear750) == '228f357ecd533b83724fc625eed37698bccb3e21e638a5c5c7c012d6a84bfc28'
assert sha(ROOT / 'linear-ref' / 'arbitration-cont-1000k.pt') == '7cd7b1c568da305c72c09b714d7f5446ff579979efd6524426e9c8ef5eb577bd'
blob = torch.load(linear750, map_location='cpu', weights_only=False)
assert blob['total_tokens'] == 749568 and blob['alpha2_frozen'] == config.CANON['alpha2_frozen']

tok, model, inj, versions, ngram_fn, FZ = studio_flow.load_stack(hf)
store = FZ.CompactPLE(str(ROOT / 'compact'))
canonical, _ = studio_flow.load_canonical_arb(ROOT, inj, arb_frozen)
del canonical
inj.set_memory(None)
model.eval()
model.requires_grad_(False)
inj.requires_grad_(False)
torch.manual_seed(1234)
arb = ResidualMLPArbitration(blob['w'], blob['b']).to('cuda')
assert arb.param_count() == 32833
assert {n for n, p in arb.named_parameters() if p.requires_grad} == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
hooks.fixed_alpha2 = config.CANON['alpha2_frozen']

full, full_man = TR.ordered_full(ROOT / 'calib')
assert full.shape[0] == 1956
for tag, blocks in (('500', 978), ('750', 1464)):
    prefix = full[:blocks].numpy().astype('<u4').tobytes()
    assert hashlib.sha256(prefix).hexdigest() == full_man['milestones'][tag]['sha256'], 'calibration prefix drift'
val = frozen_data.load_blocks(ROOT / 'frozen-v1' / 'tokens-full.uint32le', config.REVS['val_full_sha256'])
domains = {d: frozen_data.load_blocks(ROOT / 'frozen-v1' / ('tokens-%s.uint32le' % d), config.REVS['domains'][d]) for d in frozen_data.DOMAINS}
hs, _ = frozen_data.hellaswag_rows(1000, hf)
lam, _ = frozen_data.lambada_rows(1000, hf)
assert len(val) == 1024 and len(hs) == len(lam) == 1000
assert all(len(v) == 64 for v in domains.values())

# A real frozen validation block must produce identical logits and alpha8 before training.
probe = val[:1].to('cuda')
mem = store.lookup(ngram_fn(probe.cpu())).to('cuda')
linear = arb_frozen.MemoryArbitration().to('cuda')
linear.w.data.copy_(blob['w'].to('cuda'))
linear.b.data.copy_(blob['b'].to('cuda'))
linear.requires_grad_(False)
hooks.close()
old = arb_frozen.ArbHooks(model, inj, linear, FZ.decoder_layers)
old.fixed_alpha2 = config.CANON['alpha2_frozen']
old.set_memory(mem)
with torch.inference_mode():
    old_logits = model(input_ids=probe, use_cache=False).logits.detach().clone()
    old_alpha = old.last_alpha8.detach().clone()
old.close()
hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
hooks.fixed_alpha2 = config.CANON['alpha2_frozen']
hooks.set_memory(mem)
with torch.inference_mode():
    new_logits = model(input_ids=probe, use_cache=False).logits
assert torch.equal(old_alpha, hooks.last_alpha8) and torch.equal(old_logits, new_logits), 'zero residual changed baseline'
print('zero-step alpha8 and logits match linear-750K bitwise', flush=True)
del old_logits, old_alpha, new_logits, mem, probe, linear

def score(tag):
    path = ROOT / ('alpha8-mlp-%s.pt' % tag)
    os.environ['PLE_ALPHA8_RAW'] = str(path)
    with torch.inference_mode():
        res = evaluate.eval_all(model, hooks, store, ngram_fn, tok, val, domains, hs, lam, time.perf_counter(), studio_flow.HARD_S)
    assert len(res['val_blocks']) == 1024
    assert len(res['hs_correct']) == len(res['lam_correct']) == len(res['lam_nlls']) == 1000
    assert all(len(rows) == 64 for rows in res['dom_blocks'].values())
    assert set(res['alpha8']) == {'full-val', 'hellaswag', 'lambada', *domains}
    assert path.exists()
    out = dict(res, total_tokens={'500':500736, '750':749568}[tag])
    (ROOT / ('eval-mlp-%s.json' % tag)).write_text(json.dumps(out))
    print('MLP %s val=%.8f domain=%.8f LAMBADA=%.8f' % (tag, out['val_nll'], out['domain_mean'], out['lambada_nll']), flush=True)
    return out

def checkpoint(tag, cursor):
    path = ROOT / ('residual-mlp-%s.pt' % tag)
    assert torch.equal(arb.w.cpu(), blob['w'].float()) and torch.equal(arb.b.cpu(), blob['b'].float())
    state = {'arb': {k:v.detach().cpu() for k,v in arb.state_dict().items()},
             'linear_750_sha256': sha(linear750), 'calibration_tokens': cursor * 512,
             'cursor_block': cursor, 'seed': 1234, 'alpha2_frozen': config.CANON['alpha2_frozen'],
             'optimizer': 'fresh AdamW per leg, lr_peak=1e-3, wd=0, warmup=32, cosine',
             'loss': 'cross-entropy + 2*(mean(alpha8)-0.125)^2'}
    digest, reopened = TR.atomic_torch_save(state, path, ('arb', 'cursor_block'))
    assert digest == sha(path) and reopened['cursor_block'] == cursor
    assert all(torch.equal(v, reopened['arb'][k]) for k,v in state['arb'].items())
    print('residual checkpoint', tag, digest, 'reopened', flush=True)
    return digest

def paired(a, b, name):
    out = bootstrap.contrast(name, a, b)
    from random import Random
    rng = Random(1234)
    dd = {d: [x['sum'] / x['n'] - y['sum'] / y['n'] for x,y in zip(a['dom_blocks'][d], b['dom_blocks'][d])] for d in sorted(domains)}
    reps = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in dd.values()) / len(dd) for _ in range(10000))
    out['domain_mean_nll'] = {'mean':sum(sum(v) / len(v) for v in dd.values()) / len(dd),
                              'lo':reps[250], 'hi':reps[9749],
                              'p':min(1.0, 2 * min(sum(x >= 0 for x in reps), sum(x <= 0 for x in reps)) / len(reps))}
    out['paired_deltas'] = {'val_nll':[x['sum'] / x['n'] - y['sum'] / y['n'] for x,y in zip(a['val_blocks'], b['val_blocks'])],
                            'domains':dd,
                            'hellaswag_acc':[x-y for x,y in zip(a['hs_correct'], b['hs_correct'])],
                            'lambada_acc':[x-y for x,y in zip(a['lam_correct'], b['lam_correct'])],
                            'lambada_nll':[x-y for x,y in zip(a['lam_nlls'], b['lam_nlls'])]}
    print(bootstrap.summarize(out), flush=True)
    return out

reports = {}
shas = {}
for tag, start, end in ((500, 0, 978), (750, 978, 1464)):
    params = list(arb.parameters())
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
            print('MLP tokens=%d ce=%.4f mean_a8=%.4f lr=%.2e' % ((block + 1) * 512, float(ce.detach()), float(a8.detach().mean()), opt.param_groups[0]['lr']), flush=True)
        assert not [n for n,p in model.named_parameters() if p._version != versions[n]], 'backbone changed'
        del logits, ce, loss, a8
    shas[str(tag)] = checkpoint(tag, end)
    reports[str(tag)] = score(str(tag))
    del opt

hooks.close()
contrasts = []
for tag in ('500', '750'):
    for linear_tag in ('749568', '1001472'):
        contrasts.append(paired(reports[tag], baseline[linear_tag], 'mlp-%s-vs-linear-%s' % (tag, linear_tag)))
contrasts.append(paired(reports['750'], reports['500'], 'mlp-750-vs-mlp-500'))
(ROOT / 'contrasts.json').write_text(json.dumps(contrasts, indent=2))
(ROOT / 'summary.json').write_text(json.dumps({'linear-750':baseline['749568'], 'linear-1000':baseline['1001472'],
                                               'mlp-500':reports['500'], 'mlp-750':reports['750']}, indent=2))
(ROOT / 'checkpoints.json').write_text(json.dumps({'source_linear_750_sha256':sha(linear750), 'residual':shas,
                                                   'calibration_500_sha256':full_man['milestones']['500']['sha256'],
                                                   'calibration_750_sha256':full_man['milestones']['750']['sha256'],
                                                   'base_reader_sha256':config.CANON['base_reader_sha256']}, indent=2))
print('RESIDUAL MLP COMPLETE', flush=True)
