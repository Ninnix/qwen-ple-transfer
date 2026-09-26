import json
from pathlib import Path

ROOT = Path(__file__).parent
BASE = json.loads((ROOT / 'train-5m/train_5m.ipynb').read_text())
CONT = json.loads((ROOT / 'train-10m/train_10m.ipynb').read_text())
PIN = json.loads((ROOT / 'frozen.json').read_text())
OUT = ROOT / 'controls'
OUT.mkdir(exist_ok=True)


def source(nb, name):
    return ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == name))


def cell(name, body):
    return {'cell_type': 'code', 'execution_count': None, 'id': name,
            'metadata': {}, 'outputs': [], 'source': body.splitlines(keepends=True)}


names = ('c-env', 'c-install', 'c-config', 'c-hashing', 'c-reader',
         'c-inject', 'c-ple', 'c-tok', 'c-load', 'c-data', 'c-valprep')
cells = [cell(name, source(BASE, name)) for name in names]
cells.append(cell('c-freeze-train', source(CONT, 'c-freeze-train')))
cells.append(cell('c-tests', source(BASE, 'c-tests')))

cache = source(BASE, 'c-compact')
cache = cache.replace('_train_toks=FROZEN_TRAIN[:5000192//C.SEQ]',
                      '_train_toks=FROZEN_TRAIN[:500224//C.SEQ]')
cache = cache.replace('    _all=torch.cat([_val, _train_toks])', '''    _domains=[]
    for _d in ('general', 'code', 'math', 'scientific', 'multilingual'):
        _path=STAGE / ('frozen-v1__tokens-%s.uint32le' % _d)
        _raw=_path.read_bytes()
        assert hashlib.sha256(_raw).hexdigest()==EVAL_MANIFEST['domains'][_d]
        _arr=_array('I'); _arr.frombytes(_raw)
        _domains.append(torch.tensor(_arr, dtype=torch.long).view(-1, C.SEQ))
    _all=torch.cat([_val, _train_toks, *_domains])''')
cache = cache.replace('compact-reader-0-5m', 'compact-controls-500k')
cache = cache.replace("'train_start':0,'train_end':5000192",
                      "'train_start':0,'train_end':500224")
cache = cache.replace('    _uniq=torch.unique(_addrs.reshape(-1))', '''    _perm=PermutedPLE(None)
    _f=_addrs.long()
    _paddrs=_perm.O+((_f-_perm.O)*_perm.A+_perm.B)%_perm.S
    _uniq=torch.unique(torch.cat([_addrs.reshape(-1), _paddrs.reshape(-1)]))''')
cache = cache.replace('_totreq=int(_addrs.numel());',
                      '_totreq=int(_addrs.numel()+_paddrs.numel());')
cache = cache.replace("'_train_toks', '_all', '_addrs', '_uniq', '_parts'",
                      "'_train_toks', '_all', '_addrs', '_paddrs', '_uniq', '_parts'")
cells.append(cell('c-compact', cache))
cells.append(cell('c-trainfn', source(BASE, 'c-trainfn')))

run = source(BASE, 'c-train-5m')
run = run[:run.index('RUN_META =')]
run = run.replace("path = WORK / ('reader-%d.safetensors' % seen)",
                  "path = WORK / ('reader-%s-%d.safetensors' % (CURRENT, seen))")
run += '''import math

RUN_META = {'target_revision': '__TARGET_REV__',
            'tokenizer_sha256': '__TOKENIZER_SHA__',
            'ple_revision': '__PLE_REV__',
            'train_stream_sha256': TRAIN_SHA,
            'train_manifest_sha256': TRAIN_MANIFEST_SHA,
            'eval_streams_sha256': sha(WORK / 'evaluation-streams.json'),
            'compact_cache_sha256': CACHE_SHA,
            'address_mapping_sha256': ADDR_SHA,
            'injection_idx': [2, 8], 'human_layers': [3, 9],
            'reader_branches': 1, 'hidden': 2048,
            'lr': C.LR, 'weight_decay': C.WD, 'gamma_init': C.GAMMA_INIT,
            'budget_tokens': 500224}
assert _cmp.meta['train_tokens_sha256'] == hashlib.sha256(FROZEN_TRAIN[:977].numpy().astype('<u4').tobytes()).hexdigest()
assert not any(p.requires_grad for p in model.parameters())
PLE_STATS = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}
means, stds = calibrate_head_stats(_cmp)
random_memory = RandomPLE(means, stds)
permuted_memory = PermutedPLE(_cmp)
for s in (random_memory, permuted_memory): s.meta = _cmp.meta
MEMORIES = {'RANDOM': random_memory, 'PERMUTED': permuted_memory, 'REAL': _cmp}

def block_scores(reader, memory, blocks):
    out = []
    with torch.inference_mode():
        for b in blocks:
            ids = b.unsqueeze(0)
            reader.set_memory(memory.lookup(addresses(ids)).to('cuda') if memory is not None else None)
            logits = model(input_ids=ids.to('cuda'), use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   ids.to('cuda')[:, 1:].reshape(-1), reduction='sum')
            out.append({'sum': float(loss), 'n': 511})
    return out

def score_condition(label, state, memory):
    reader = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
    if state is not None:
        reader.load_state_dict({k: v.to('cuda') for k, v in state.items()})
    full = block_scores(reader, memory, val_full)
    domains = {}
    for d in ('general', 'code', 'math', 'scientific', 'multilingual'):
        path = STAGE / ('frozen-v1__tokens-%s.uint32le' % d)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == EVAL_MANIFEST['domains'][d]
        a = __import__('array').array('I'); a.frombytes(raw)
        blocks = torch.tensor(a, dtype=torch.long).view(-1, 512)
        domains[d] = block_scores(reader, memory, blocks)
    reader.close()
    report = {'full': full, 'domains': domains,
              'full_nll': sum(x['sum'] for x in full)/sum(x['n'] for x in full),
              'domain_nll': {d: sum(x['sum'] for x in rows)/sum(x['n'] for x in rows)
                             for d, rows in domains.items()}}
    (WORK / ('control-eval-%s.json' % label)).write_text(json.dumps(report))
    print('CONTROL_EVAL', label, report['full_nll'], report['domain_nll'], flush=True)
    return report

CURRENT = 'DISABLED'
disabled = score_condition('DISABLED', None, None)
results = {'DISABLED': disabled}
for CURRENT, memory in MEMORIES.items():
    RUN_META['memory_condition'] = CURRENT
    run = train_reader([2, 8], 1, 500000, memory, val_full, CURRENT.lower(),
                       ckpts=[500000], baselines={'fast': {'loss': disabled['full_nll']}})
    seen = run['checkpoints'][-1]['tokens']
    assert seen == 500224
    path = WORK / ('reader-%s-500224.safetensors' % CURRENT)
    state = load_file(str(path), device='cpu')
    results[CURRENT] = score_condition(CURRENT, state, memory)
    assert PLE_STATS == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in _cstore.part_paths.values()}

def paired(a, b):
    from random import Random
    dd = [x['sum']/x['n'] - y['sum']/y['n'] for x,y in zip(a,b)]
    rng = Random(1234); n=len(dd)
    reps = sorted(sum(dd[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {'delta': sum(dd)/n, 'lo': reps[250], 'hi': reps[9749], 'n': n}

comparisons = {}
for a,b in (('REAL','PERMUTED'), ('REAL','RANDOM'), ('REAL','DISABLED'),
            ('PERMUTED','RANDOM'), ('PERMUTED','DISABLED'), ('RANDOM','DISABLED')):
    key = a + '_minus_' + b
    comparisons[key] = {'full': paired(results[a]['full'], results[b]['full']),
                        'domains': {d: paired(results[a]['domains'][d], results[b]['domains'][d])
                                    for d in results[a]['domains']}}
summary = {'budget_tokens': 500224,
           'full_nll': {k: v['full_nll'] for k,v in results.items()},
           'domains': {k: v['domain_nll'] for k,v in results.items()},
           'comparisons': comparisons,
           'train_stream_sha256': TRAIN_SHA,
           'cache_sha256': CACHE_SHA, 'mapping_sha256': ADDR_SHA}
(WORK / 'controls-500k-summary.json').write_text(json.dumps(summary, indent=2))
print('CONTROLS_COMPLETE', json.dumps(summary), flush=True)
'''
run = run.replace('__TARGET_REV__', PIN['target_revision']).replace('__TOKENIZER_SHA__', PIN['target_tokenizer_sha256']).replace('__PLE_REV__', PIN['source_revision'])
cells.append(cell('c-controls', run))

nb = {'cells': cells, 'metadata': BASE['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
(OUT / 'controls_500k.ipynb').write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / 'train-5m/kernel-metadata.json').read_text())
meta.update(code_file='controls_500k.ipynb', id='ninnix/qwengram-2b-controls-500k-t4',
            title='Qwengram 2B controls 500K T4',
            dataset_sources=meta['dataset_sources'] + ['ninnix/qwengram-2b-checkpoints'])
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
