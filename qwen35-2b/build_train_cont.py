import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STAGE = int(sys.argv[1])
assert STAGE in (10, 15)
PREV = 5 if STAGE == 10 else 10
START = 5000192 if STAGE == 10 else 10000384
END = 10000384 if STAGE == 10 else 15000064
THRESHOLD = STAGE * 1000000
ROOT_OUT = ROOT / ('train-%dm' % STAGE)
ROOT_OUT.mkdir(exist_ok=True)
nb = json.loads((ROOT / 'train-5m/train_5m.ipynb').read_text())
pin = json.loads((ROOT / 'frozen.json').read_text())


def source(name):
    return ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == name))


def put(name, body):
    c = next(c for c in nb['cells'] if c.get('id') == name)
    c['source'] = body.splitlines(keepends=True)
    c['execution_count'] = None
    c['outputs'] = []


trainer = source('c-trainfn')
old = "'tok_s':round(seen/(time.perf_counter()-t0),1)"
assert trainer.count(old) == 1
trainer = trainer.replace(old, "'tok_s':round((seen-_seen0)/(time.perf_counter()-t0),1)")
trainer = trainer.replace("'rss_MiB':round(rss_mb(),1)",
                          "'rss_MiB':round(rss_mb(),1),'wall_s':round(time.perf_counter()-t0,1)")
needle = '        seen=step*C.SEQ; iloss+=loss.item()*(C.SEQ-1); icount+=C.SEQ-1\n'
assert trainer.count(needle) == 1
trainer = trainer.replace(needle, needle + '''        if step % 1000 == 0:
            print('TRAIN_PROGRESS', seen, round((seen-_seen0)/(time.perf_counter()-t0),1), flush=True)
''')
put('c-trainfn', trainer)


freeze = source('c-freeze-train')
start = freeze.index('stream = FineWebEduStream(')
end = freeze.index("print('frozen reader stream'", start)
freeze = freeze[:start] + '''from array import array
train_hits = list(Path('/kaggle/input').rglob('reader-train-15m.uint32le'))
assert len(train_hits) == 1
TRAIN_PATH = train_hits[0]
TRAIN_MANIFEST_PATH = TRAIN_PATH.with_suffix('.json')
assert TRAIN_MANIFEST_PATH.exists()
TRAIN_MANIFEST = json.loads(TRAIN_MANIFEST_PATH.read_text())
assert TRAIN_MANIFEST['target_revision'] == '%s'
assert TRAIN_MANIFEST['tokenizer_sha256'] == '%s'
assert TRAIN_MANIFEST['token_count'] == 15000064
TRAIN_SHA = TRAIN_MANIFEST['sha256']
assert sha(TRAIN_PATH) == TRAIN_SHA
TRAIN_MANIFEST_SHA = sha(TRAIN_MANIFEST_PATH)
tokens = array('I')
tokens.frombytes(TRAIN_PATH.read_bytes())
FROZEN_TRAIN = torch.tensor(tokens, dtype=torch.long).view(-1, C.SEQ)
assert FROZEN_TRAIN.shape == (29297, 512)
print('reopened frozen reader stream', TRAIN_SHA, flush=True)
''' % (pin['target_revision'], pin['target_tokenizer_sha256']) + freeze[end:]
put('c-freeze-train', freeze)

cache = source('c-compact')
cache = cache.replace('_train_toks=FROZEN_TRAIN[:5000192//C.SEQ]',
                      '_train_toks=FROZEN_TRAIN[%d//C.SEQ:%d//C.SEQ]' % (START, END))
cache = cache.replace('compact-reader-0-5m', 'compact-reader-%d-%dm' % (PREV, STAGE))
cache = cache.replace("'train_start':0,'train_end':5000192",
                      "'train_start':%d,'train_end':%d" % (START, END))
put('c-compact', cache)

run = source('c-train-5m')
run = run[:run.index('RUN_META =')]
run += '''RUN_META = {'target_revision': '%s', 'tokenizer_sha256': '%s',
            'ple_revision': '%s', 'train_stream_sha256': TRAIN_SHA,
            'train_manifest_sha256': TRAIN_MANIFEST_SHA,
            'eval_streams_sha256': sha(WORK / 'evaluation-streams.json'),
            'compact_cache_sha256': CACHE_SHA, 'address_mapping_sha256': ADDR_SHA,
            'injection_idx': [2, 8], 'human_layers': [3, 9],
            'reader_branches': 1, 'hidden': 2048,
            'lr': C.LR, 'weight_decay': C.WD, 'gamma_init': C.GAMMA_INIT,
            'schedule': {'mode': 'continuation', 'peak': 1.5e-5,
                         'warm_steps': 200, 'horizon_tokens': %d}}
assert not any(p.requires_grad for p in model.parameters())
assert _cmp.meta['train_tokens_sha256'] == hashlib.sha256(FROZEN_TRAIN[%d//512:%d//512].numpy().astype('<u4').tobytes()).hexdigest()
PLE_STATS = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
summary_hits = list(Path('/kaggle/input').rglob('reader-%dm-summary.json'))
assert len(summary_hits) == 1
previous = json.loads(summary_hits[0].read_text())
resume_path = summary_hits[0].parent / 'reader-real-%dm-%d.pt'
assert sha(resume_path) == previous['checkpoint_sha256']
resume = torch.load(resume_path, map_location='cpu', weights_only=False)
assert resume['seen'] == %d
assert resume['provenance']['train_stream_sha256'] == TRAIN_SHA
assert resume['cfg'] == ((2, 8), 1)
assert len(resume['optimizer']['state']) == 8
base = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, 0.0).to('cuda')
stock_fast = eval_loss(base, None, val_fast, False)
''' % (pin['target_revision'], pin['target_tokenizer_sha256'], pin['source_revision'],
       THRESHOLD, START, END, PREV, PREV, START, START)
if STAGE == 15:
    run += "stock_full = eval_loss(base, None, val_full, False)\n"
run += '''base.close()
baselines = {'fast': stock_fast%s}
out = train_reader([2, 8], 1, %d, _cmp, val_fast, 'real-%dm',
                   ckpts=[%d], baselines=baselines,%s
                   resume=resume, cont_peak=1.5e-5, cont_warm=200)
seen = out['checkpoints'][-1]['tokens']
assert seen == %d
ck = WORK / ('reader-real-%dm-%%d.pt' %% seen)
verified = torch.load(ck, map_location='cpu', weights_only=False)
assert verified['seen'] == seen and verified['provenance']['train_stream_sha256'] == TRAIN_SHA
assert PLE_STATS == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
summary = {'stock_fast': stock_fast,%s
           'reader': out['checkpoints'][-1],
           'reader_sha256': sha(WORK / ('reader-%%d.safetensors' %% seen)),
           'checkpoint_sha256': sha(ck), 'train_sha256': TRAIN_SHA,
           'cache_sha256': CACHE_SHA, 'address_sha256': ADDR_SHA}
(WORK / 'reader-%dm-summary.json').write_text(json.dumps(summary, indent=2))
print('READER_%dM_COMPLETE', json.dumps(summary), flush=True)
''' % (", 'full': stock_full" if STAGE == 15 else '', THRESHOLD, STAGE,
       THRESHOLD, "val_full=val_full, full_at={%d}," % THRESHOLD if STAGE == 15 else '',
       END, STAGE, "'stock_full': stock_full," if STAGE == 15 else '', STAGE, STAGE)
put('c-train-5m', run)

meta = json.loads((ROOT / 'train-5m/kernel-metadata.json').read_text())
meta.update(code_file='train_%dm.ipynb' % STAGE,
            id='ninnix/qwengram-2b-reader-%d-%dm-t4' % (PREV, STAGE),
            title='Qwengram 2B reader %d %dM T4' % (PREV, STAGE),
            dataset_sources=meta['dataset_sources'] + ['ninnix/qwengram-2b-checkpoints'])
(ROOT_OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
(ROOT_OUT / meta['code_file']).write_text(json.dumps(nb, indent=1))
