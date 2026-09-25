import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE = Path('/kaggle/input/qwen-ple-arb-cont-750k-stage')
ROOT = Path('/kaggle/working/ple-cont')
CODE = ROOT / 'code'
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
assert torch.cuda.is_available()
torch.zeros(1, device='cuda')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

if not (STAGE / 'transfer-shas.json').exists():
    mounts = sorted(str(p) for p in Path('/kaggle/input').iterdir())
    hits = list(Path('/kaggle/input').rglob('transfer-shas.json'))
    print('input mounts:', mounts, 'manifest hits:', [str(p) for p in hits], flush=True)
    assert len(hits) == 1, 'frozen stage dataset unavailable'
    STAGE = hits[0].parent
manifest = json.loads((STAGE / 'transfer-shas.json').read_text())
for name, meta in manifest.items():
    source = STAGE / name
    assert source.stat().st_size == meta['size'] and sha(source) == meta['sha256'], name + ' input drift'
    group, base = name.split('__', 1)
    dest = ROOT / ('code' if group == 'code' else group) / base
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    assert sha(dest) == meta['sha256'], name + ' copy drift'
print('frozen stage verified:', len(manifest), 'files', flush=True)
prior_hits = list(Path('/kaggle/input').rglob('prior-eval-shas.json'))
assert len(prior_hits) <= 1
if prior_hits:
    prior_dir = prior_hits[0].parent
    prior = json.loads(prior_hits[0].read_text())
    assert prior['source_kernel'] == 'ninnix/qwen-ple-idx8-arbiter-continuation-p100/3'
    for name, meta in prior['files'].items():
        source = prior_dir / name
        assert source.stat().st_size == meta['size'] and sha(source) == meta['sha256'], name + ' prior eval drift'
        group, base = name.split('__', 1) if '__' in name else ('', name)
        dest = ROOT / group / base
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        assert sha(dest) == meta['sha256'], name + ' prior eval copy drift'
    print('P100 canonical/500/750 evaluations recovered and SHA verified', flush=True)

# Preserve the original scoring path and persist its already-collected gate values.
ev = CODE / 'evaluate.py'
src = ev.read_text()
needle = '    stats = {k: alpha_stats(torch.cat(vs)) for k, vs in collect.items() if vs}\n'
assert src.count(needle) == 1
src = src.replace(needle, "    raw = __import__('os').environ.get('PLE_ALPHA8_RAW')\n    if raw: torch.save({k: [v.cpu() for v in vs] for k, vs in collect.items()}, raw)\n" + needle)
ev.write_text(src)
sys.path.insert(0, str(CODE))
import bootstrap
import frozen_generated as FZ
import studio_flow

def checkpoint(tag, expected, tokens):
    path = ROOT / 'milestones' / ('arbitration-cont-%dk.pt' % tag)
    assert sha(path) == expected, '%dK checkpoint SHA mismatch' % tag
    blob = torch.load(path, map_location='cpu', weights_only=False)
    assert blob['total_tokens'] == tokens and blob['cursor_block'] == tokens // 512
    assert tuple(blob['w'].shape) == (1024,) and blob['b'].numel() == 1
    assert blob['alpha2_frozen'] == 1.267012596130371
    assert blob['optimizer_resume_from_v49'] is False
    print('checkpoint %dK SHA/reopen ok %s' % (tag, expected[:16]), flush=True)
    return blob

checkpoint(500, '31c9f09dfef29ec322cfb6e4fe0290720f6ec64445aae0311918630865bff87f', 500736)
checkpoint(750, '228f357ecd533b83724fc625eed37698bccb3e21e638a5c5c7c012d6a84bfc28', 749568)

tok, model, inj, versions, ngram_fn, FZ = studio_flow.load_stack(hf)
store = FZ.CompactPLE(str(ROOT / 'compact'))

def complete(rep):
    assert len(rep['val_blocks']) == 1024
    assert len(rep['hs_correct']) == len(rep['lam_correct']) == len(rep['lam_nlls']) == 1000
    assert set(rep['dom_blocks']) == {'general', 'code', 'math', 'scientific', 'multilingual'}
    assert all(len(rows) == 64 for rows in rep['dom_blocks'].values())
    assert set(rep['alpha8']) == set(rep['dom_blocks']) | {'full-val', 'hellaswag', 'lambada'}
    return rep

def eval_arm(tag):
    studio_flow.T0 = time.perf_counter()
    os.environ['PLE_ALPHA8_RAW'] = str(ROOT / 'milestones' / ('alpha8-%s.pt' % tag))
    if tag == 'canonical':
        rep = studio_flow.eval_canonical_c(tok, model, inj, versions, ngram_fn, FZ, store)
    else:
        rep = studio_flow.eval_single(tok, model, inj, versions, ngram_fn, FZ, store, tag)
    complete(rep)
    assert Path(os.environ['PLE_ALPHA8_RAW']).exists(), 'gate values missing'
    print('%s val=%.8f domain=%.8f LAMBADA=%.8f' % (tag, rep['val_nll'], rep['domain_mean'], rep['lambada_nll']), flush=True)
    return rep

# Reuse completed P100 per-item scores when a prior run stopped before training.
if prior_hits:
    assert json.loads((ROOT / 'qualification.json').read_text())['status'] == 'passed'
    canonical = complete(json.loads((ROOT / 'canonical-c-eval.json').read_text()))
    e500 = complete(json.loads((ROOT / 'milestones' / 'eval-500k.json').read_text()))
    e750 = complete(json.loads((ROOT / 'milestones' / 'eval-750k.json').read_text()))
else:
    old = ROOT / 'milestones' / 'eval-500k.json'
    old.rename(ROOT / 'milestones' / 'eval-500k-lightning.json')
    canonical = eval_arm('canonical')
    e500 = eval_arm(500)
    e750 = eval_arm(750)
from config import ALPHA8_REF, QUAL_TOL
ref = ALPHA8_REF['full-val']
for key in ('mean', 'std', 'p05', 'p50', 'p95'):
    tol = QUAL_TOL['alpha8_q'] if key.startswith('p') else QUAL_TOL['alpha8_' + key]
    assert abs(canonical['alpha8']['full-val'][key] - ref[key]) <= tol, 'canonical gate drift: ' + key
assert abs(canonical['val_nll'] - 2.865383) <= QUAL_TOL['val_nll'], 'canonical NLL drift'
assert abs(e500['val_nll'] - 2.865274439469242) <= QUAL_TOL['val_nll'], '500K NLL drift'

def contrast(a, b, label):
    c = bootstrap.contrast(label, a, b)
    from random import Random
    rng = Random(1234)
    domains = sorted(a['dom_blocks'])
    dd = {d: [x['sum'] / x['n'] - y['sum'] / y['n'] for x, y in zip(a['dom_blocks'][d], b['dom_blocks'][d])] for d in domains}
    reps = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in dd.values()) / len(dd) for _ in range(10000))
    c['domain_mean_nll'] = {'mean':sum(sum(v) / len(v) for v in dd.values()) / len(dd),
                            'lo':reps[250], 'hi':reps[9749],
                            'p':min(1.0, 2.0 * min(sum(x >= 0 for x in reps), sum(x <= 0 for x in reps)) / len(reps)),
                            'n_domains':len(dd), 'blocks_per_domain':64}
    c['paired_deltas'] = {'val_nll':[x['sum'] / x['n'] - y['sum'] / y['n'] for x, y in zip(a['val_blocks'], b['val_blocks'])],
                          'domains':dd,
                          'hellaswag_acc':[x - y for x, y in zip(a['hs_correct'], b['hs_correct'])],
                          'lambada_acc':[x - y for x, y in zip(a['lam_correct'], b['lam_correct'])],
                          'lambada_nll':[x - y for x, y in zip(a['lam_nlls'], b['lam_nlls'])]}
    print(bootstrap.summarize(c), flush=True)
    print('  five-domain mean: d=%+.8f [%.8f, %.8f]' % (c['domain_mean_nll']['mean'], c['domain_mean_nll']['lo'], c['domain_mean_nll']['hi']), flush=True)
    return c

first = [contrast(e500, canonical, '500736-vs-canonical319488'),
         contrast(e750, e500, '749568-vs-500736')]
if prior_hits:
    saved = json.loads((ROOT / 'contrasts-through-750.json').read_text())
    for new, old in zip(first, saved):
        assert new['name'] == old['name']
        for key in ('val_nll', 'lambada_nll', 'domain_mean_nll'):
            assert abs(new[key]['mean'] - old[key]['mean']) < 1e-12, 'prior contrast drift: ' + key
(ROOT / 'contrasts-through-750.json').write_text(json.dumps(first, indent=2))

studio_flow.T0 = time.perf_counter()
res = studio_flow.train_legs(tok, model, inj, versions, ngram_fn, FZ, store, limit_blocks=1956)
assert res['stopped'] == 'done' and res['cursor'] == 1956, res
million = ROOT / 'milestones' / 'arbitration-cont-1000k.pt'
million_sha = sha(million)
checkpoint(1000, million_sha, 1001472)
(ROOT / 'milestones' / 'checkpoint-1000k-sha.json').write_text(json.dumps({'file':million.name, 'sha256':million_sha, 'total_tokens':1001472}, indent=2))
e1000 = eval_arm(1000)

contrasts = first + [contrast(e1000, e750, '1001472-vs-749568'),
                     contrast(e1000, canonical, '1001472-vs-canonical319488')]
(ROOT / 'contrasts.json').write_text(json.dumps(contrasts, indent=2))
(ROOT / 'summary.json').write_text(json.dumps({'canonical': canonical, '500736':e500, '749568':e750, '1001472':e1000}, indent=2))
print('ALL MILESTONES COMPLETE', flush=True)
