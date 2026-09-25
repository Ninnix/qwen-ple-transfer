# Continue the verified REAL R=1 IDX2+IDX8 reader from 15M to 20M.
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

OUT = Path('/kaggle/working/reader-scale-20m')
OUT.mkdir(parents=True, exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def pre_eval_save_reader(inj, seen, threshold):
    path = OUT / ('real-r1-%d.reader.safetensors' % seen)
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
    print('PRE-EVAL reader milestone', threshold, seen, sha, flush=True)

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
assert len(cur['optimizer']['state']) == 8 and 'scheduler' in cur
frozen15 = load_file(str(hits[0].parent / weights_name), device='cpu')
assert set(frozen15) == set(cur['reader'])
assert all(torch.equal(frozen15[k], cur['reader'][k]) for k in frozen15)
print('15M reader resume SHA/reopen and published weights verified', flush=True)

val_fast, mf = load_validation('fast')
val_full, mF = load_validation('full')
assert mf['tokens_sha256'] == '4c702b4313fca2fa7fc5ffaf242912ab48a1564fecda52dd01982e2b756e4abd'
assert mF['tokens_sha256'] == '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c'
assert mF['dataset_rev'] == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
meta = json.loads((_cdir / 'compact.json').read_text())
assert meta['token_train'] == 20000256 and meta['token_val'] == 524288
assert meta['dataset_rev'] == mF['dataset_rev'] and meta['train_skip_docs'] == mF['skip_docs']
store = _cmp

probe = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
base_fast = eval_loss(probe, None, val_fast, False)
base_full = eval_loss(probe, None, val_full, False)
probe.close()
assert abs(base_fast['loss'] - 2.895546885152619) < 1e-4
assert abs(base_full['loss'] - 2.905584982696578) < 1e-4
baselines = {'fast':base_fast, 'full':base_full}
print('frozen disabled baselines', base_fast['loss'], base_full['loss'], flush=True)

run = train_reader([2, 8], 1, 20000000, store, val_fast, 'real-scale-20m-r1',
                   ckpts=[20000000], val_full=val_full, baselines=baselines,
                   full_at={20000000}, resume=cur, cont_peak=1.5e-5, cont_warm=200)
fin = run['checkpoints'][-1]
assert fin['tokens'] == 20000256 and fin['threshold'] == 20000000 and fin['val_set'] == 'full'
raw = Path('/kaggle/working/reader-real-scale-20m-r1-20000256.pt')
raw_sha = digest(raw)
ck = torch.load(raw, map_location='cpu', weights_only=False)
assert ck['seen'] == 20000256 and ck['cfg'] == ((2, 8), 1)
assert len(ck['optimizer']['state']) == 8
pre = OUT / 'real-r1-20000256.reader.safetensors'
weights = load_file(str(pre), device='cpu')
assert set(weights) == set(ck['reader'])
assert all(torch.equal(weights[k], ck['reader'][k]) for k in weights)
print('full reader resume SHA/reopen', raw_sha, flush=True)

run_meta = {
    'tag':'real-20m-r1', 'memory_condition':'real', 'tokens':20000256,
    'threshold':20000000, 'val_set':'full', 'placement_idx':[2, 8], 'branches':1,
    'seed':C.SEED, 'lr':C.LR, 'wd':C.WD, 'warmup_frac':C.WARMUP_FRAC,
    'gamma_init':C.GAMMA_INIT, 'target_rev':mF['target_rev'],
    'ple_revision':meta['ple_revision'], 'dataset_rev':mF['dataset_rev'],
    'train_skip_docs':mF['skip_docs'], 'val_fast_sha256':mf['tokens_sha256'],
    'val_full_sha256':mF['tokens_sha256'],
    'lr_schedule':{'mode':'continuation', 'peak':1.5e-5, 'warm_steps':200,
                   'horizon_tokens':20000000, 'base_lr':C.LR},
    'resume_parent_sha256':source['files'][resume_name]['sha256'],
    'origin':'verified 15M optimizer continuation',
}
metrics = {k:fin.get(k) for k in ('train_loss', 'val_loss', 'dval', 'ppl', 'gamma', 'gate', 'tok_s', 'peak_GiB')}
art = Path('/kaggle/working/qwen35-08b-ple-artifacts')
art.mkdir(parents=True, exist_ok=True)
bundle = write_bundle(art, 'real-20m-r1', ck, run_meta, metrics)
bundle_sha = {p.name:digest(p) for p in bundle.iterdir() if p.is_file()}
assert all(torch.equal(load_file(str(bundle / 'reader.safetensors'), device='cpu')[k], v) for k,v in weights.items())
reopened = torch.load(bundle / 'resume.pt', map_location='cpu', weights_only=False)
assert reopened['seen'] == 20000256 and len(reopened['optimizer']['state']) == 8
assert all(torch.equal(reopened['reader'][k], v) for k,v in weights.items())
summary = {'tokens':20000256, 'threshold':20000000,
           'full_val_raw_reader':fin['val_loss'], 'dval':fin['dval'],
           'gamma':fin['gamma'], 'gate':fin['gate'],
           'pre_eval_reader_sha256':digest(pre), 'raw_resume_sha256':raw_sha,
           'bundle_sha256':bundle_sha,
           'resume_parent_sha256':source['files'][resume_name]['sha256'],
           'reader_source':source['source_kernel']}
(OUT / 'reader-20m-summary.json').write_text(json.dumps(summary, indent=2))
print('READER 20M COMPLETE', summary, flush=True)
import shutil
shutil.copyfile(_cdir / 'compact.json', OUT / 'compact-20m.json')
shutil.rmtree(_cdir)
print('released completed 20M compact cache after checkpoint verification', flush=True)
