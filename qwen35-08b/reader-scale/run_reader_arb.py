import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/kaggle/working/reader-arb')
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

readers_manifest = find('reader-stage-shas.json')
readers = json.loads(readers_manifest.read_text())
assert readers['source_kernel'] == 'ninnix/qwen-ple-reader-scale-7p5m-10m-p100/3'
for name, meta in readers['files'].items():
    src = readers_manifest.parent / name
    assert src.stat().st_size == meta['size'] and sha(src) == meta['sha256'], name
    dst = ROOT / 'readers' / name
    dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == meta['sha256'], name
print('7.5M and 10M reader milestones SHA verified', flush=True)

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
store = FZ.CompactPLE(str(ROOT / 'compact'))
val = frozen_data.load_blocks(ROOT / 'frozen-v1' / 'tokens-full.uint32le', config.REVS['val_full_sha256'])
domains = {d:frozen_data.load_blocks(ROOT / 'frozen-v1' / ('tokens-%s.uint32le' % d), config.REVS['domains'][d]) for d in frozen_data.DOMAINS}
hs, _ = frozen_data.hellaswag_rows(1000, hf)
lam, _ = frozen_data.lambada_rows(1000, hf)
assert len(val) == 1024 and len(hs) == len(lam) == 1000 and all(len(v) == 64 for v in domains.values())
full, full_man = TR.ordered_full(ROOT / 'calib')
for tag, blocks in (('500', 978), ('750', 1464)):
    assert hashlib.sha256(full[:blocks].numpy().astype('<u4').tobytes()).hexdigest() == full_man['milestones'][tag]['sha256']

def load_reader(path, expected):
    assert sha(path) == expected
    state = load_file(str(path), device='cpu')
    assert set(state) == set(inj.state_dict())
    inj.load_state_dict({k:v.to('cuda') for k,v in state.items()})
    inj.requires_grad_(False)
    inj.set_memory(None)
    return {n:float(p.detach()) for n,p in inj.named_parameters() if n.endswith('gamma')}

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

linear750 = ROOT / 'milestones' / 'arbitration-cont-750k.pt'
assert sha(linear750) == '228f357ecd533b83724fc625eed37698bccb3e21e638a5c5c7c012d6a84bfc28'
load_reader(ROOT / 'ckpts' / config.CANON['base_reader_file'], config.CANON['base_reader_sha256'])
blob = torch.load(linear750, map_location='cpu', weights_only=False)
arb = arb_frozen.MemoryArbitration().to('cuda')
arb.w.data.copy_(blob['w'].to('cuda'))
arb.b.data.copy_(blob['b'].to('cuda'))
arb.requires_grad_(False)
baseline = score('reader5m-linear750', arb)
reference = json.loads((ROOT / 'readers' / 'linear-750-eval.json').read_text())
for key in ('val_nll', 'domain_mean', 'lambada_nll', 'hs_acc', 'lambada_acc'):
    assert abs(baseline[key] - reference[key]) < 1e-5, '5M baseline drift: ' + key
print('5M balanced baseline reproduces prior P100 evaluation', flush=True)

def checkpoint(label, arb, reader_sha, tokens):
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

def paired(a, b, name):
    out = bootstrap.contrast(name, a, b)
    from random import Random
    rng = Random(1234)
    dd = {d:[x['sum']/x['n'] - y['sum']/y['n'] for x,y in zip(a['dom_blocks'][d], b['dom_blocks'][d])] for d in sorted(domains)}
    reps = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v)/len(v) for v in dd.values())/len(dd) for _ in range(10000))
    out['domain_mean_nll'] = {'mean':sum(sum(v)/len(v) for v in dd.values())/len(dd),
                              'lo':reps[250], 'hi':reps[9749],
                              'p':min(1.0, 2*min(sum(x>=0 for x in reps), sum(x<=0 for x in reps))/len(reps))}
    out['paired_deltas'] = {'val_nll':[x['sum']/x['n']-y['sum']/y['n'] for x,y in zip(a['val_blocks'],b['val_blocks'])],
                            'domains':dd,
                            'hellaswag_acc':[x-y for x,y in zip(a['hs_correct'],b['hs_correct'])],
                            'lambada_acc':[x-y for x,y in zip(a['lam_correct'],b['lam_correct'])],
                            'lambada_nll':[x-y for x,y in zip(a['lam_nlls'],b['lam_nlls'])]}
    print(bootstrap.summarize(out), flush=True)
    return out

reports = {'reader5m-linear750':baseline}
checkpoints = {}
gamma = {}
for reader_tag in ('7p5m', '10m'):
    reader_path = ROOT / 'readers' / ('real-%s-r1.reader.safetensors' % reader_tag)
    reader_sha = readers['files'][reader_path.name]['sha256']
    gamma[reader_tag] = load_reader(reader_path, reader_sha)
    torch.manual_seed(1234)
    arb = arb_frozen.MemoryArbitration().to('cuda')
    arb.alpha2_raw.requires_grad_(False)
    assert torch.count_nonzero(arb.w) == 0
    assert abs(float(arb.alpha8(torch.zeros(1, 1, 1024, device='cuda'))) - 0.125) < 1e-7
    hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
    hooks.fixed_alpha2 = config.CANON['alpha2_frozen']
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
                print(reader_tag, 'arb tokens', (block + 1)*512, 'ce', float(ce.detach()),
                      'mean alpha8', float(a8.detach().mean()), flush=True)
            assert not [n for n,p in model.named_parameters() if p._version != versions[n]]
            del logits, ce, a8, loss
        label = 'reader%s-arb%d' % (reader_tag, tag)
        checkpoints[label] = checkpoint(label, arb, reader_sha, end*512)
        hooks.close()
        arb.requires_grad_(False)
        reports[label] = score(label, arb)
        if tag == 500:
            arb.w.requires_grad_(True)
            arb.b.requires_grad_(True)
            hooks = arb_frozen.ArbHooks(model, inj, arb, FZ.decoder_layers)
            hooks.fixed_alpha2 = config.CANON['alpha2_frozen']
        del opt

contrasts = []
for reader_tag in ('7p5m', '10m'):
    for tag in (500, 750):
        label = 'reader%s-arb%d' % (reader_tag, tag)
        contrasts.append(paired(reports[label], baseline, label + '-vs-reader5m-linear750'))
    contrasts.append(paired(reports['reader%s-arb750' % reader_tag], reports['reader%s-arb500' % reader_tag],
                            'reader%s-arb750-vs-arb500' % reader_tag))
contrasts.append(paired(reports['reader10m-arb750'], reports['reader7p5m-arb750'],
                        'reader10m-arb750-vs-reader7p5m-arb750'))
(ROOT / 'summary.json').write_text(json.dumps(reports, indent=2))
(ROOT / 'contrasts.json').write_text(json.dumps(contrasts, indent=2))
(ROOT / 'checkpoints.json').write_text(json.dumps({'reader_source':readers['source_kernel'],
                                                   'reader_shas':{k:readers['files']['real-%s-r1.reader.safetensors'%k]['sha256'] for k in ('7p5m','10m')},
                                                   'arbitration_shas':checkpoints, 'gamma':gamma,
                                                   'alpha2_frozen':config.CANON['alpha2_frozen'],
                                                   'calibration_shas':{k:full_man['milestones'][k]['sha256'] for k in ('500','750')},
                                                   'frozen_eval_shas':config.REVS}, indent=2))
print('READER-SCALE ARBITRATION COMPLETE', flush=True)
