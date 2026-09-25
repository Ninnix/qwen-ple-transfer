# Warm-start four shared-value reader branches from the verified 15M R=1 checkpoint.
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

OUT = Path('/kaggle/working/reader-r4-early')
OUT.mkdir(parents=True, exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pre_eval_save_reader(inj, seen, threshold):
    path = OUT / ('real-r4-%d.reader.safetensors' % seen)
    tmp = path.with_suffix('.safetensors.tmp')
    state = {k:v.detach().cpu() for k,v in inj.state_dict().items()}
    save_file(state, str(tmp))
    with open(tmp, 'rb+') as f:
        f.flush()
        os.fsync(f.fileno())
    reopened = load_file(str(tmp), device='cpu')
    assert set(reopened) == set(state)
    assert all(torch.equal(reopened[k], v) for k,v in state.items())
    os.replace(tmp, path)
    sha = digest(path)
    assert all(torch.equal(load_file(str(path), device='cpu')[k], v) for k,v in state.items())
    print('PRE-EVAL R4 reader milestone', threshold, seen, sha, flush=True)

hits = list(Path('/kaggle/input').rglob('reader-15m-shas.json'))
assert len(hits) == 1, hits
source = json.loads(hits[0].read_text())
assert source['source_kernel'] == 'ninnix/qwen-ple-real-r1-reader-scale-15m-p100/1'
resume_name = 'real-15m-r1.resume.pt'
weights_name = 'real-15m-r1.reader.safetensors'
for name in (resume_name, weights_name):
    path = hits[0].parent / name
    record = source['files'][name]
    assert path.stat().st_size == record['size'] and digest(path) == record['sha256']
assert source['files'][resume_name]['sha256'] == '7d9369cd7a1c0bdacd3604746138a09b8fd2e5d7e9ae8212507ee4976cd86229'
assert source['files'][weights_name]['sha256'] == 'e4a760163ec07568178ab48aa533235a9af878183caf120d4e29ce1c8ce4b9dc'
cur = torch.load(hits[0].parent / resume_name, map_location='cpu', weights_only=False)
assert cur['seen'] == 15000064 and cur['cfg'] == ((2, 8), 1)
frozen15 = load_file(str(hits[0].parent / weights_name), device='cpu')
assert set(frozen15) == set(cur['reader'])
assert all(torch.equal(frozen15[k], cur['reader'][k]) for k in frozen15)
print('15M reader SHA/reopen and optimizer parent verified', flush=True)

val_fast, mf = load_validation('fast')
val_full, mF = load_validation('full')
assert mf['tokens_sha256'] == '4c702b4313fca2fa7fc5ffaf242912ab48a1564fecda52dd01982e2b756e4abd'
assert mF['tokens_sha256'] == '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c'
assert mF['dataset_rev'] == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
meta = json.loads((_cdir / 'compact.json').read_text())
assert meta['train_start'] == cur['seen'] and meta['train_end'] == 16000000
assert meta['token_train'] == 999936 and meta['token_val'] == 524288
assert meta['dataset_rev'] == mF['dataset_rev'] and meta['train_skip_docs'] == mF['skip_docs']
store = _cmp

g = torch.Generator().manual_seed(C.SEED)
r4 = {}
for name, value in frozen15.items():
    if name.endswith('.keys.0.weight'):
        noise = [torch.randn(value.shape, generator=g) for _ in range(2)]
        noise = [x * (value.std() * 1e-4 / x.std()) for x in noise]
        for b, delta in enumerate((noise[0], -noise[0], noise[1], -noise[1])):
            r4[name.replace('.keys.0.weight', '.keys.%d.weight' % b)] = value + delta
    elif name.endswith('.beta'):
        r4[name] = value.repeat(4)
    else:
        r4[name] = value.clone()
assert all(torch.equal(r4[n], v) for n,v in frozen15.items() if n.endswith(('.value.weight', '.gamma')))
assert all(torch.equal(r4[n], v.repeat(4)) for n,v in frozen15.items() if n.endswith('.beta'))

torch.manual_seed(C.SEED)
h = torch.randn(1, 32, C.HIDDEN)
m = torch.randn(1, 32, C.MEM_DIM)
direct = {}
with torch.inference_mode():
    for site in (2, 8):
        one = SharedValueReader(C.MEM_DIM, C.HIDDEN, 1)
        four = SharedValueReader(C.MEM_DIM, C.HIDDEN, 4)
        one.load_state_dict({k.split('.', 2)[2]:v for k,v in frozen15.items() if k.startswith('readers.%d.' % site)})
        four.load_state_dict({k.split('.', 2)[2]:v for k,v in r4.items() if k.startswith('readers.%d.' % site)})
        direct[str(site)] = float((one(h,m) - four(h,m)).abs().max())
        assert direct[str(site)] <= 1e-5, direct
print('R4 warm-start direct reader max|delta|', direct, flush=True)

ids = val_fast[0].unsqueeze(0).cuda()
memory = store.lookup(addresses(ids.cpu())).cuda()
with torch.inference_mode():
    probe = ReaderInjection(model, [2,8], C.MEM_DIM, C.HIDDEN, 1).cuda()
    probe.load_state_dict({k:v.cuda() for k,v in frozen15.items()})
    probe.set_memory(memory)
    logits1 = model(input_ids=ids, use_cache=False).logits.cpu()
    probe.close()
    probe = ReaderInjection(model, [2,8], C.MEM_DIM, C.HIDDEN, 4).cuda()
    probe.load_state_dict({k:v.cuda() for k,v in r4.items()})
    probe.set_memory(memory)
    logits4 = model(input_ids=ids, use_cache=False).logits.cpu()
    probe.close()
logit_delta = float((logits1.float() - logits4.float()).abs().max())
# A one-ULP fp16 hidden difference can spread through the frozen decoder.
assert logit_delta <= 0.05, logit_delta
print('R4 warm-start frozen block max|logit delta|', logit_delta, flush=True)
del probe, logits1, logits4, memory, ids, one, four, h, m

probe = ReaderInjection(model, [2,8], C.MEM_DIM, C.HIDDEN, 4).cuda()
optimizer = torch.optim.AdamW(probe.parameters(), lr=C.LR, weight_decay=C.WD)
warm = {'reader':r4, 'optimizer':optimizer.state_dict(), 'seen':cur['seen'],
        'torch_rng':cur['torch_rng'], 'cuda_rng':cur['cuda_rng'],
        'init_state':{k:v.clone() for k,v in r4.items()}}
probe.close()
del optimizer, probe, cur
baselines = {'fast':{'loss':2.895546885152619}, 'full':{'loss':2.905584982696578}}
run = train_reader([2,8], 4, 16000000, store, val_fast, 'real-r4-early',
                   ckpts=[16000000], val_full=val_full, baselines=baselines,
                   full_at={16000000}, resume=warm, cont_peak=1.5e-5, cont_warm=200)
fin = run['checkpoints'][-1]
assert fin['tokens'] == 16000000 and fin['threshold'] == 16000000 and fin['val_set'] == 'full'
raw = Path('/kaggle/working/reader-real-r4-early-16000000.pt')
raw_sha = digest(raw)
ck = torch.load(raw, map_location='cpu', weights_only=False)
assert ck['seen'] == 16000000 and ck['cfg'] == ((2,8),4)
assert len(ck['optimizer']['state']) == 14
pre = OUT / 'real-r4-16000000.reader.safetensors'
weights = load_file(str(pre), device='cpu')
assert set(weights) == set(ck['reader'])
assert all(torch.equal(weights[k], ck['reader'][k]) for k in weights)
print('R4 full resume SHA/reopen', raw_sha, flush=True)

run_meta = {'tag':'real-r4-early','memory_condition':'real','tokens':16000000,
            'additional_tokens':999936,'threshold':16000000,'placement_idx':[2,8],
            'branches':4,'seed':C.SEED,'lr':C.LR,'wd':C.WD,
            'gamma_init':'copied from 15M R1','key_init':'copied with paired +/- 1e-4 relative perturbations',
            'value_init':'exact 15M R1 copy','optimizer':'fresh AdamW for expanded reader',
            'lr_schedule':{'mode':'continuation','peak':1.5e-5,'warm_steps':200,'horizon_tokens':16000000},
            'target_rev':mF['target_rev'],'ple_revision':meta['ple_revision'],
            'dataset_rev':mF['dataset_rev'],'train_skip_docs':mF['skip_docs'],
            'train_tokens_sha256':meta['train_tokens_sha256'],
            'compact_addrs_sha256':meta['addrs_sha256'],'compact_rows_sha256':meta['rows_sha256'],
            'val_fast_sha256':mf['tokens_sha256'],'val_full_sha256':mF['tokens_sha256'],
            'reader_parent_sha256':source['files'][weights_name]['sha256'],
            'resume_parent_sha256':source['files'][resume_name]['sha256'],
            'warm_start_direct_max_abs':direct,'warm_start_logit_max_abs':logit_delta}
metrics = {k:fin.get(k) for k in ('train_loss','val_loss','dval','ppl','gamma','gate','tok_s','peak_GiB')}
art = Path('/kaggle/working/qwen35-08b-ple-artifacts')
art.mkdir(parents=True, exist_ok=True)
bundle = write_bundle(art, 'real-r4-early', ck, run_meta, metrics)
bundle_sha = {p.name:digest(p) for p in bundle.iterdir() if p.is_file()}
assert all(torch.equal(load_file(str(bundle/'reader.safetensors'), device='cpu')[k], v) for k,v in weights.items())
reopened = torch.load(bundle/'resume.pt', map_location='cpu', weights_only=False)
assert reopened['seen'] == 16000000 and len(reopened['optimizer']['state']) == 14
assert all(torch.equal(reopened['reader'][k], v) for k,v in weights.items())
summary = {'tokens':16000000,'additional_tokens':999936,'full_val_raw_reader':fin['val_loss'],
           'dval':fin['dval'],'pre_eval_reader_sha256':digest(pre),'raw_resume_sha256':raw_sha,
           'bundle_sha256':bundle_sha,'warm_start_direct_max_abs':direct,
           'warm_start_logit_max_abs':logit_delta,'resume_parent_sha256':source['files'][resume_name]['sha256']}
(OUT/'reader-r4-summary.json').write_text(json.dumps(summary, indent=2))
shutil.copyfile(_cdir/'compact.json', OUT/'compact-r4.json')
print('R4 EARLY MILESTONE COMPLETE', summary, flush=True)
shutil.rmtree(_cdir)
