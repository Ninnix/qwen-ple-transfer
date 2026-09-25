# REAL R=1 IDX2+IDX8 reader continuation from the verified 5M resume.
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

OUT = Path('/kaggle/working/reader-scale')
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

resume_hits = list(Path('/kaggle/input').rglob('real-5m-r1.resume.pt'))
assert len(resume_hits) == 1, resume_hits
resume_path = resume_hits[0]
resume_sha = 'fb1fd83a93eef95bf41f98dc7145619cffacac7dadc9376ab797abdd0450902f'
assert digest(resume_path) == resume_sha
cur = torch.load(resume_path, map_location='cpu', weights_only=False)
assert cur['seen'] == 5000192 and cur['cfg'] == ((2, 8), 1)
assert len(cur['optimizer']['state']) == 8 and 'scheduler' in cur
reader5 = resume_path.parent / 'real-5m-r1.reader.safetensors'
assert digest(reader5) == '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594'
frozen5 = load_file(str(reader5), device='cpu')
assert set(frozen5) == set(cur['reader'])
assert all(torch.equal(frozen5[k], cur['reader'][k]) for k in frozen5)
print('5M reader resume SHA/reopen and published weights verified', flush=True)

val_fast, mf = load_validation('fast')
val_full, mF = load_validation('full')
assert mf['tokens_sha256'] == '4c702b4313fca2fa7fc5ffaf242912ab48a1564fecda52dd01982e2b756e4abd'
assert mF['tokens_sha256'] == '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c'
assert mF['dataset_rev'] == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
meta = json.loads((_cdir / 'compact.json').read_text())
assert meta['token_train'] == 10000384 and meta['token_val'] == 524288
assert meta['dataset_rev'] == mF['dataset_rev'] and meta['train_skip_docs'] == mF['skip_docs']
store = _cmp

probe = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
base_fast = eval_loss(probe, None, val_fast, False)
base_full = eval_loss(probe, None, val_full, False)
probe.close()
assert abs(base_fast['loss'] - 2.895519334042842) < 0.02
assert abs(base_full['loss'] - 2.9055815991123595) < 0.02
baselines = {'fast':base_fast, 'full':base_full}
print('frozen disabled baselines', base_fast['loss'], base_full['loss'], flush=True)

ART = Path('/kaggle/working/qwen35-08b-ple-artifacts')
ART.mkdir(parents=True, exist_ok=True)
def run_metadata(label, expected, threshold, parent):
    return {
        'tag':'real-' + label + '-r1', 'memory_condition':'real', 'tokens':expected,
        'threshold':threshold, 'val_set':'full', 'placement_idx':[2, 8], 'branches':1,
        'seed':C.SEED, 'lr':C.LR, 'wd':C.WD, 'warmup_frac':C.WARMUP_FRAC,
        'gamma_init':C.GAMMA_INIT, 'target_rev':mF['target_rev'],
        'ple_revision':meta['ple_revision'], 'dataset_rev':mF['dataset_rev'],
        'train_skip_docs':mF['skip_docs'], 'val_fast_sha256':mf['tokens_sha256'],
        'val_full_sha256':mF['tokens_sha256'],
        'lr_schedule':{'mode':'continuation', 'peak':1.5e-5, 'warm_steps':200,
                       'horizon_tokens':threshold, 'base_lr':C.LR},
        'resume_parent_sha256':parent, 'origin':'verified optimizer continuation',
    }

stage_hits = list(Path('/kaggle/input').rglob('reader-7p5m-stage.json'))
assert len(stage_hits) == 1, stage_hits
stage = json.loads(stage_hits[0].read_text())
assert stage['source_kernel'] == 'ninnix/qwen-ple-reader-scale-7p5m-10m-p100/2'
assert stage['resume_parent_sha256'] == resume_sha
for name, record in stage['files'].items():
    path = stage_hits[0].parent / name
    assert path.stat().st_size == record['size'] and digest(path) == record['sha256']
cur = torch.load(stage_hits[0].parent / 'reader-real-scale-r1-7500288.pt', map_location='cpu', weights_only=False)
assert cur['seen'] == 7500288 and cur['cfg'] == ((2, 8), 1)
assert len(cur['optimizer']['state']) == 8
reader7 = load_file(str(stage_hits[0].parent / 'real-r1-7500288.reader.safetensors'), device='cpu')
assert set(reader7) == set(cur['reader'])
assert all(torch.equal(reader7[k], cur['reader'][k]) for k in reader7)
fin7 = stage['metrics']
assert fin7['tokens'] == 7500288 and fin7['val_set'] == 'full'
bundle7 = write_bundle(ART, 'real-7p5m-r1', cur,
                       run_metadata('7p5m', 7500288, 7500000, resume_sha), fin7)
bundle7_sha = {p.name:digest(p) for p in bundle7.iterdir() if p.is_file()}
assert all(torch.equal(load_file(str(bundle7 / 'reader.safetensors'), device='cpu')[k], v) for k,v in reader7.items())
assert torch.load(bundle7 / 'resume.pt', map_location='cpu', weights_only=False)['seen'] == 7500288
results = {'7p5m':{'tokens':7500288, 'threshold':7500000, 'full_val_raw_reader':fin7['val_loss'],
                  'dval':fin7['dval'], 'gamma':fin7['gamma'], 'gate':fin7['gate'],
                  'pre_eval_reader_sha256':stage['files']['real-r1-7500288.reader.safetensors']['sha256'],
                  'raw_resume_sha256':stage['files']['reader-real-scale-r1-7500288.pt']['sha256'],
                  'bundle_sha256':bundle7_sha, 'resume_parent_sha256':resume_sha}}
(OUT / 'reader-scale-summary.json').write_text(json.dumps(results, indent=2))
print('7.5M reader recovered, SHA/reopen verified and bundled', flush=True)
previous_sha = stage['files']['reader-real-scale-r1-7500288.pt']['sha256']
for threshold, label, expected in ((10000000, '10m', 10000384),):
    assert cur['seen'] < expected
    run = train_reader([2, 8], 1, threshold, store, val_fast, 'real-scale-r1',
                       ckpts=[threshold], val_full=val_full, baselines=baselines,
                       full_at={threshold}, resume=cur, cont_peak=1.5e-5, cont_warm=200)
    fin = run['checkpoints'][-1]
    assert fin['tokens'] == expected and fin['threshold'] == threshold and fin['val_set'] == 'full'
    path = Path('/kaggle/working/reader-real-scale-r1-%d.pt' % expected)
    ck_sha = digest(path)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['seen'] == expected and ck['cfg'] == ((2, 8), 1)
    assert len(ck['optimizer']['state']) == 8
    pre = OUT / ('real-r1-%d.reader.safetensors' % expected)
    weights = load_file(str(pre), device='cpu')
    assert set(weights) == set(ck['reader'])
    assert all(torch.equal(weights[k], ck['reader'][k]) for k in weights)
    print('full reader resume SHA/reopen', label, ck_sha, flush=True)
    run_meta = run_metadata(label, expected, threshold, previous_sha)
    metrics = {k:fin.get(k) for k in ('train_loss', 'val_loss', 'dval', 'ppl', 'gamma', 'gate', 'tok_s', 'peak_GiB')}
    bundle = write_bundle(ART, 'real-' + label + '-r1', ck, run_meta, metrics)
    bundle_sha = {p.name:digest(p) for p in bundle.iterdir() if p.is_file()}
    assert all(torch.equal(load_file(str(bundle / 'reader.safetensors'), device='cpu')[k], v) for k,v in weights.items())
    reopened = torch.load(bundle / 'resume.pt', map_location='cpu', weights_only=False)
    assert reopened['seen'] == expected and all(torch.equal(reopened['reader'][k], v) for k,v in weights.items())
    results[label] = {'tokens':expected, 'threshold':threshold, 'full_val_raw_reader':fin['val_loss'],
                      'dval':fin['dval'], 'gamma':fin['gamma'], 'gate':fin['gate'],
                      'pre_eval_reader_sha256':digest(pre), 'raw_resume_sha256':ck_sha,
                      'bundle_sha256':bundle_sha, 'resume_parent_sha256':previous_sha}
    (OUT / 'reader-scale-summary.json').write_text(json.dumps(results, indent=2))
    print('reader milestone complete', label, results[label], flush=True)
    cur = ck
    previous_sha = ck_sha

print('READER SCALE 7.5M AND 10M COMPLETE', flush=True)
