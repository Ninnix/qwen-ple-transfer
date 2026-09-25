# Stage A2-arb-route: inference-only causal routing diagnostic on frozen REAL-5M R=1 (IDX 2+8).
# No training, no weight changes, no R4. Uses deployed-B gate outputs only.
# Question: is routing the same memory budget to uncertain positions causally better than to easy ones?
# Arms (IDX2 fixed 1.25 everywhere; only IDX8 placement changes, per-sequence multiset exact):
#   learned  = B deployment as-is (must reproduce arb-B numbers: invariant below)
#   shuffled = per-sequence pinned permutation of alpha8 (same multiset, location destroyed)
#   hard     = largest alpha8 to highest disabled-entropy scored positions (leftovers to unscored)
#   easy     = largest alpha8 to lowest disabled-entropy scored positions (leftovers to unscored)
# Unscored positions (block tail, prompt prefix) always receive the smallest leftovers in order.
# Entropy comes from the disabled pure-backbone pass of the same sequence.
# Eval: full-val NLL (1024 blocks), hellaswag acc (1000), lambada acc/NLL (1000), 5 frozen domains (64 each).
# Persists: routing.json, hellaswag-<arm>.jsonl, lambada-<arm>.jsonl, config-route.json, timings-route.json.
import hashlib, json, math, time
import torch, torch.nn.functional as F
from pathlib import Path
from torch import nn
T0 = time.perf_counter()
EOUT = Path('/kaggle/working/eval-stageA2-arb-route'); EOUT.mkdir(parents=True, exist_ok=True)
EFI_MOUNT = Path('/kaggle/input/qwen-ple-reader-checkpoints')
ARB_MOUNT = Path('/kaggle/input/qwen-ple-a2-arb')
RDIR = Path('/kaggle/input/qwen-ple-a2-v34')
DLDIR = Path('/kaggle/working/ckpt-dl-arb-route'); DLDIR.mkdir(parents=True, exist_ok=True)
assert RDIR.exists(), 'results dataset missing: attach ninnix/qwen-ple-a2-v34'
assert ARB_MOUNT.exists(), 'arb dataset missing: attach ninnix/qwen-ple-a2-arb'
REF = {
    'target_rev': '2fc06364715b967f1860aea9cf38778875588b17',
    'source_rev_prefix': '236dfdf28582',
    'ple_revision': '236dfdf285828023ca3bcd3f37366c58a3469b13',
    'ckpt_sha': {'real-5m': '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594'},
    'ckpt_tokens': {'real-5m': 5000192},
    'arb_sha': {
        'arbitration-b.pt': '595d195a84c65acd2aa414d61726e080b54801b4676b3d7a2256ca8a7bcd9948',
        'arbitration-c.pt': '2cbdd7913413c87ed45500d4496540fcbab30e3a97d6070a54e5f38d94eef3d9',
        'arbitration-b.json': '0331a2e7b281a7469448c87edee67260cd8462412b486c6e56b9ea8b75f85230',
        'arbitration-c.json': '89bf8edeb7d2e26dfb87cef15015615e8d364222971b49fbda074bdcd020035a',
        'alpha8-stats.json': '0793e0b6899e6d27b7c37397c0374c33383477a631a83117760e031d8be9183a',
    },
    'mc': {
        'hellaswag': {'dataset': 'Rowan/hellaswag', 'rev': '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'n': 1000},
        'lambada': {'dataset': 'EleutherAI/lambada_openai', 'rev': '900124bf3b8235c6daf21033af9948b3f07346c4', 'n': 1000},
    },
    'val_full_sha256': '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c',
    'domains': {
        'general': 'a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5',
        'code': '735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421',
        'math': 'c27f42f5f69de1aaf26020c7e14c2c141f8fa8490c5d95952c06e1c5bffa6648',
        'scientific': 'c33632cf96170ca7ae92d4013702c14097fd79bb069c76e40fb3a531091936d8',
        'multilingual': '65a26da5419aaa32520219d9d1c5909769a7a19a88ff16e85df7d69c38028a55',
    },
}
B_REF = {'val_nll': 2.865490686403562, 'hs_acc': 0.404, 'lam_acc': 0.453, 'lam_nll': 2.253693464072421,
         'general': 3.148797033350995, 'code': 1.4981278458686724, 'math': 1.4137548299218343,
         'scientific': 2.256194297581503, 'multilingual': 3.661386721288155}
ARMS = ('learned', 'shuffled', 'hard', 'easy')
assert trev == REF['target_rev'], 'tokenizer revision drift: %s' % trev
assert srev.startswith(REF['source_rev_prefix']), 'source revision drift'
print('frozen revisions ok (tokenizer/model/ple)', flush=True)
print('route: 4 arms, per-sequence budget exact, inference only; no R4, no weight updates', flush=True)
timings = {}
def _mark(name):
    timings[name] = round(time.perf_counter() - T0, 1)
    print('[t=%ds] %s' % (timings[name], name), flush=True)
def _guard():
    assert (time.perf_counter() - T0) <= C.EVAL_HARD_S - 1800, 'HARD-GUARD trip'
def _file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
for _f, _h in REF['arb_sha'].items():
    assert (ARB_MOUNT / _f).exists(), 'missing arb file: ' + _f
    assert _file_sha(ARB_MOUNT / _f) == _h, 'arb bytes mismatch: ' + _f
print('arb outputs pinned (B weights + stats SHAs match)', flush=True)
def _fetch_ckpt():
    need = ['protocol.json', 'real-5m-r1.reader.safetensors', 'real-5m-r1.run.json']
    if EFI_MOUNT.exists() and all((EFI_MOUNT / f).exists() for f in need):
        print('checkpoints: mounted dataset (schema-validated)', flush=True)
        return EFI_MOUNT
    import os, subprocess, sys
    _env = dict(os.environ); _env['KAGGLE_API_TOKEN'] = secret_value_1
    for _f in need:
        _r = subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'ninnix/qwen-ple-reader-checkpoints', '-f', _f, '-p', str(DLDIR), '--force'], capture_output=True, text=True, env=_env, timeout=1200)
        assert _r.returncode == 0 and (DLDIR / _f).exists(), 'checkpoint download failed: ' + _f
    print('checkpoints: api download (mount missing)', flush=True)
    return DLDIR
CKDIR = _fetch_ckpt()
_proto = json.loads((CKDIR / 'protocol.json').read_text())
assert _proto.get('ple_revision') == REF['ple_revision'], 'PLE revision drift'
def _load_sfc(pt_name, run_name, expect_tokens, expect_sha):
    from safetensors.torch import load_file
    run = json.loads((CKDIR / run_name).read_text())
    assert run.get('token_count', run.get('tokens')) == expect_tokens, 'budget mismatch: ' + run_name
    h = _file_sha(CKDIR / pt_name)
    assert h == expect_sha, 'bytes mismatch: ' + pt_name
    return load_file(str(CKDIR / pt_name), device='cpu'), h
st5, _h5 = _load_sfc('real-5m-r1.reader.safetensors', 'real-5m-r1.run.json', REF['ckpt_tokens']['real-5m'], REF['ckpt_sha']['real-5m'])
print('frozen REAL-5M R=1 weights ready', flush=True)
_mark('checkpoints ready')
V = {}
for _f in ['hellaswag.jsonl', 'lambada.jsonl']:
    assert (RDIR / _f).exists(), 'missing frozen input: ' + _f
V['rows'] = {}
for _t in ['hellaswag', 'lambada']:
    V['rows'][_t] = [json.loads(x) for x in (RDIR / f'{_t}.jsonl').read_text().splitlines() if x.strip()]
print('frozen v34 MC rows loaded', flush=True)
hs_rows, hs_meta = _hellaswag_rows(C.EVAL_MC_N)
lam_rows, lam_meta = _lambada_rows(C.EVAL_MC_N)
for _name, _rows, _meta in [('hellaswag', hs_rows, hs_meta), ('lambada', lam_rows, lam_meta)]:
    _ref = REF['mc'][_name]
    assert _meta['dataset'] == _ref['dataset'] and str(_meta['rev']) == _ref['rev'], 'dataset drift: ' + _name
    assert len(_rows) == _ref['n'] == len(V['rows'][_name]), 'row count drift: ' + _name
print('MC manifests identical to v34 (hellaswag+lambada)', flush=True)
val_full_chk, mfull_chk = load_validation('full')
assert mfull_chk['tokens_sha256'] == REF['val_full_sha256'], 'frozen full-val drift'
dom_blocks = {}
for _d in ('general', 'code', 'math', 'scientific', 'multilingual'):
    _b, _m = build_domain_blocks(tokenizer, _d, C.EVAL_DOMAIN_TOKENS)
    assert _m['tokens_sha256'] == REF['domains'][_d], 'domain token drift: ' + _d
    assert _b.shape[0] == 64, 'domain block drift: ' + _d
    dom_blocks[_d] = _b
print('frozen full-val (1024 blocks) + 5 domains ok', flush=True)
_mark('frozen inputs verified')
tokenizer.padding_side = 'left'
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = C.EOS
inj = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
inj.load_state_dict({k: v.to('cuda') for k, v in st5.items()})
inj.requires_grad_(False)
model.requires_grad_(False)
assert sum(1 for p in inj.parameters() if p.requires_grad) == 0, 'reader must stay grad-free'
assert sum(1 for p in model.parameters() if p.requires_grad) == 0, 'backbone must stay frozen'
orig_g2 = float(inj.readers['2'].gamma.detach().cpu())
orig_g8 = float(inj.readers['8'].gamma.detach().cpu())
assert abs(orig_g2 - 0.06166713312268257) < 1e-6 and abs(orig_g8 - 0.07537073642015457) < 1e-6, '5M weights mismatch'
print('frozen reader ok: IDX2=%.6f IDX8=%.6f' % (orig_g2, orig_g8), flush=True)
class GateOnly(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = w.float()
        self.b = float(b)
    def alpha8(self, h8):
        hn = rms_norm(h8.detach().float())
        return 0.5 * torch.sigmoid((hn * self.w.to(hn.device)).sum(-1) + self.b)
stB = torch.load(str(ARB_MOUNT / 'arbitration-b.pt'), map_location='cpu', weights_only=True)
assert tuple(stB['w'].shape) == (1024,), 'arb w shape drift'
gB = GateOnly(stB['w'], stB['b'])
print('deployed-B gate loaded (frozen, inference only)', flush=True)
SHUF = torch.Generator().manual_seed(1234)
class RouteHooks(nn.Module):
    # Deployed-B reproduction with swappable IDX8 placement. IDX2 fixed 1.25 always.
    # mode off = pure pass-through (disabled baseline). Otherwise w8 is rearranged alpha8.
    # scored_mask marks gate positions that predict a scored target; leftovers go to the rest.
    def __init__(self, model, frozen, gate):
        super().__init__()
        self.gate = gate
        self.fr2 = frozen.readers['2']
        self.fr8 = frozen.readers['8']
        self.mode = 'off'
        self.memory = None
        self.ent = None
        self.scored = None
        self.applied = {}
        dec = decoder_layers(model)
        self.handles = [
            dec[2].register_forward_pre_hook(self._hook2(), with_kwargs=True),
            dec[8].register_forward_pre_hook(self._hook8(), with_kwargs=True),
        ]
    def _place(self, a8):
        T = a8.shape[0]
        if self.mode == 'learned':
            return a8
        if self.mode == 'shuffled':
            return a8[torch.randperm(T, generator=SHUF)]
        desc = torch.argsort(a8, descending=True)
        sc = torch.nonzero(self.scored).flatten()
        un = torch.nonzero(~self.scored).flatten()
        order = torch.argsort(self.ent[sc], descending=(self.mode == 'hard'))
        w = torch.empty_like(a8)
        w[sc[order]] = a8[desc[:len(sc)]]
        w[un] = a8[desc[len(sc):]]
        return w
    def _hook2(self):
        def fn(mod, args, kw):
            if self.mode == 'off' or self.memory is None:
                return args, kw
            h = args[0] if args else kw['hidden_states']
            m = self.memory if self.memory.shape[1] == h.shape[1] else self.memory[:, :h.shape[1]]
            aug = self.fr2(h, m)
            out = h + 1.25 * (aug - h)
            if args:
                return (out, *args[1:]), kw
            kw['hidden_states'] = out
            return args, kw
        return fn
    def _hook8(self):
        def fn(mod, args, kw):
            if self.mode == 'off' or self.memory is None:
                return args, kw
            h = args[0] if args else kw['hidden_states']
            m = self.memory if self.memory.shape[1] == h.shape[1] else self.memory[:, :h.shape[1]]
            a8 = self.gate.alpha8(h).reshape(-1)
            w = self._place(a8)
            _s, _n = self.applied.get(self.mode, (0.0, 0))
            self.applied[self.mode] = (_s + float(w.sum()), _n + int(w.numel()))
            aug = self.fr8(h, m)
            out = h + w.to(h.dtype).unsqueeze(-1) * (aug - h)
            if args:
                return (out, *args[1:]), kw
            kw['hidden_states'] = out
            return args, kw
        return fn
    def set_memory(self, m):
        self.memory = m
    def close(self):
        [h.remove() for h in self.handles]; self.handles.clear()
def _build_compact_route(pdir, seqs):
    import hashlib as _hl
    from array import array as _array
    from safetensors import safe_open as _so
    _real = []
    with torch.inference_mode():
        for _s in seqs:
            _real.append(ngram_indices(_s.unsqueeze(0).cpu()).reshape(-1))
    uniq = torch.unique(torch.cat(_real)).long().cpu()
    print('route union addresses: %d' % len(uniq), flush=True)
    del _real
    _uh = _hl.sha256(_array('I', uniq.tolist()).tobytes()).hexdigest()
    _ms = MountPLE()
    assert _ms.ple_revision == REF['ple_revision'], 'PLE revision drift'
    _uq = uniq.sort().values
    pdir.mkdir(parents=True, exist_ok=True)
    _af = (pdir / 'addrs.u32').open('wb'); _rf = (pdir / 'rows.u8').open('wb')
    _ha = _hl.sha256(); _hr = _hl.sha256(); _n = 0; _prev = -1
    _pa = torch.div(_uq, C.ROWS_PER_PART, rounding_mode='floor'); _lo = _uq % C.ROWS_PER_PART
    _bf = {}
    for _p in torch.unique(_pa).tolist():
        _bf.setdefault(str(_ms.part_paths[_p]), []).append(_p)
    for _fp in sorted(_bf):
        with _so(_fp, framework='pt', device='cpu') as _fh:
            for _p in sorted(_bf[_fp]):
                _full = _fh.get_slice(MountPLE.tensor_name(_p))[:]
                _pos = torch.nonzero(_pa == _p).flatten()
                _au = _uq.index_select(0, _pos)
                assert int(_au[0]) > _prev; _prev = int(_au[-1])
                _ab = _array('I', _au.tolist()).tobytes()
                _rb = bytes(_full.index_select(0, _lo.index_select(0, _pos)).view(torch.uint8).flatten().tolist())
                _af.write(_ab); _rf.write(_rb); _ha.update(_ab); _hr.update(_rb); _n += _pos.numel()
    _af.close(); _rf.close()
    (pdir / 'compact.json').write_text(json.dumps({'format': 'qwen-ple-compact-a2-arb-route', 'version': 1, 'address_count': _n, 'uniq_sha256': _uh, 'row_dim': C.ROW_DIM, 'scale': float(_ms.scale), 'ple_revision': _ms.ple_revision, 'addrs_sha256': _ha.hexdigest(), 'rows_sha256': _hr.hexdigest()}, indent=2))
    _cc = CompactPLE(pdir)
    _g = torch.Generator().manual_seed(0)
    _samp = _uq[torch.randint(0, len(_uq), (2048,), generator=_g)].reshape(128, 16)
    assert (_cc.lookup(_samp) - _ms.lookup(_samp)).abs().max().item() == 0.0
    return _cc
_seq_all = [val_full_chk[i] for i in range(val_full_chk.shape[0])]
for _d in dom_blocks:
    for _bi in range(dom_blocks[_d].shape[0]):
        _seq_all.append(dom_blocks[_d][_bi])
for r in hs_rows:
    p = tokenizer(r['ctx'], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    for e in r['endings']:
        _seq_all.append(torch.cat([p, tokenizer((' ' + e.strip() if not e.startswith(' ') else e), return_tensors='pt', add_special_tokens=False)['input_ids'][0]])[:512])
for r in lam_rows:
    _seq_all.append(torch.cat([tokenizer(r['context'], return_tensors='pt', add_special_tokens=False)['input_ids'][0], tokenizer((' ' + r['target'].strip() if not r['target'].startswith(' ') else r['target']), return_tensors='pt', add_special_tokens=False)['input_ids'][0]])[:512])
_cgr = _build_compact_route(WORK / 'compact-a2-arb-route', _seq_all)
del _seq_all
_mark('compact ready (route union)')
rt = RouteHooks(model, inj, gB)
CHUNK = 128
def _disabled_forward(ids):
    rt.mode = 'off'
    rt.memory = None
    rt.ent = None
    rt.scored = None
    inj.set_memory(None)
    with torch.inference_mode():
        return model(input_ids=ids.to('cuda'), use_cache=False).logits
def _routed_forward(ids, mode, ent, scored):
    rt.mode = mode
    rt.ent = ent
    rt.scored = scored
    with torch.inference_mode():
        rt.set_memory(_cgr.lookup(addresses(ids.cpu())).to('cuda'))
        return model(input_ids=ids.to('cuda'), use_cache=False).logits
def _full_entropy(lg_cpu, n_tgt):
    ents = []
    for _s in range(0, n_tgt, CHUNK):
        _e = min(_s + CHUNK, n_tgt)
        lp = torch.log_softmax(lg_cpu[_s:_e], dim=-1)
        pr = lp.exp()
        ents.append((-pr * lp).sum(-1))
        del lp, pr
    return torch.cat(ents)
_dp = hs_rows[0]
_d1a = _disabled_forward(torch.cat([tokenizer(_dp['ctx'], return_tensors='pt', add_special_tokens=False)['input_ids'][0], tokenizer(' ' + _dp['endings'][0], return_tensors='pt', add_special_tokens=False)['input_ids'][0]])[:512].unsqueeze(0))
_d1b = _disabled_forward(torch.cat([tokenizer(_dp['ctx'], return_tensors='pt', add_special_tokens=False)['input_ids'][0], tokenizer(' ' + _dp['endings'][0], return_tensors='pt', add_special_tokens=False)['input_ids'][0]])[:512].unsqueeze(0))
assert torch.equal(_d1a.cpu(), _d1b.cpu()), 'eval path not deterministic'
print('eval determinism ok', flush=True)
del _d1a, _d1b, _dp
_mark('scoring ready')
import random as _r
def _boot_diff(vals_a, vals_b, n_boot=10000, seed=1234):
    dd = [a - b for a, b in zip(vals_a, vals_b)]
    rng = _r.Random(seed)
    n = len(dd)
    reps = sorted(sum(dd[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    ge = sum(1 for x in reps if x >= 0.0)
    le = sum(1 for x in reps if x <= 0.0)
    return {'mean': sum(dd) / n, 'lo': reps[int(0.025 * n_boot)], 'hi': reps[int(0.975 * n_boot) - 1], 'p': min(1.0, 2.0 * min(ge, le) / n_boot)}
def _block_nll_all(blocks):
    sums = {a: 0.0 for a in ARMS}
    ns = {a: 0 for a in ARMS}
    for _bi in range(blocks.shape[0]):
        _guard()
        b = blocks[_bi]
        L = int(b.shape[0])
        lg_d = _disabled_forward(b.unsqueeze(0)).float().cpu()[0]
        ent = _full_entropy(lg_d, L - 1)
        scored = torch.ones(L, dtype=torch.bool)
        scored[-1] = False
        for _arm in ARMS:
            lg = _routed_forward(b.unsqueeze(0), _arm, ent, scored).float()
            s = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b[1:].reshape(-1).to(lg.device), reduction='sum').item()
            sums[_arm] += s
            ns[_arm] += L - 1
            del lg
        del lg_d, ent
        if (_bi + 1) % 256 == 0:
            print('blocks %d/%d' % (_bi + 1, blocks.shape[0]), flush=True)
    return {_a: sums[_a] / max(1, ns[_a]) for _a in ARMS}
def _hs_all():
    base = V['rows']['hellaswag']
    outs = {a: [] for a in ARMS}
    accs = {a: [] for a in ARMS}
    for _j, r in enumerate(hs_rows):
        _guard()
        assert r['label'] == base[_j]['label'], 'prompt drift: hellaswag idx %d' % _j
        p = tokenizer(r['ctx'], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        opts = []
        for e in r['endings']:
            t = tokenizer(e if e.startswith(' ') else ' ' + e, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            opts.append(torch.cat([p, t], dim=0))
        lps = {a: [] for a in ARMS}
        for _o, _ids in enumerate(opts):
            L = int(_ids.shape[0])
            lg_d = _disabled_forward(_ids.unsqueeze(0)).float().cpu()[0]
            ent = _full_entropy(lg_d, L - 1)
            t = tokenizer(r['endings'][_o] if r['endings'][_o].startswith(' ') else ' ' + r['endings'][_o], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            scored = torch.zeros(L, dtype=torch.bool)
            scored[max(0, L - 1 - int(t.shape[0])):L - 1] = True
            del lg_d
            for _arm in ARMS:
                lg = _routed_forward(_ids.unsqueeze(0), _arm, ent, scored).float()[0]
                lp = torch.log_softmax(lg[p.shape[0] - 1:L - 1], dim=-1)
                lps[_arm].append(lp.gather(1, _ids[p.shape[0]:L].to(lg.device).unsqueeze(1)).sum().item())
                del lg, lp
            del ent
        for _arm in ARMS:
            pred = max(range(len(opts)), key=lambda i: lps[_arm][i])
            rec = dict(base[_j])
            rec[_arm] = {'logprobs': lps[_arm], 'pred': pred, 'correct': int(pred == r['label'])}
            outs[_arm].append(rec)
            accs[_arm].append(int(pred == r['label']))
        if (_j + 1) % 250 == 0:
            print('hs %d/1000' % (_j + 1), flush=True)
    return outs, {_a: sum(accs[_a]) / max(1, len(accs[_a])) for _a in ARMS}, accs
def _lam_all():
    base = V['rows']['lambada']
    outs = {a: [] for a in ARMS}
    accs = {a: [] for a in ARMS}
    nlls = {a: [] for a in ARMS}
    for _j, r in enumerate(lam_rows):
        _guard()
        assert r['target'] == base[_j]['target'], 'prompt drift: lambada idx %d' % _j
        p = tokenizer(r['context'], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        t = tokenizer(r['target'] if r['target'].startswith(' ') else ' ' + r['target'], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        _ids = torch.cat([p, t], dim=0)
        L = int(_ids.shape[0])
        lg_d = _disabled_forward(_ids.unsqueeze(0)).float().cpu()[0]
        ent = _full_entropy(lg_d, L - 1)
        scored = torch.zeros(L, dtype=torch.bool)
        scored[max(0, L - 1 - int(t.shape[0])):L - 1] = True
        del lg_d
        for _arm in ARMS:
            lg = _routed_forward(_ids.unsqueeze(0), _arm, ent, scored).float()[0]
            lp = torch.log_softmax(lg[p.shape[0] - 1:L - 1], dim=-1)
            tt = _ids[p.shape[0]:L].to(lg.device)
            nll = -lp.gather(1, tt.unsqueeze(1)).sum().item()
            acc = int(bool((lp.argmax(-1).to('cpu') == tt.to('cpu')).all()))
            rec = dict(base[_j])
            rec[_arm] = {'nll': nll, 'nll_per_tok': nll / max(1, int(tt.shape[0])), 'correct': acc, 'ntok': int(tt.shape[0])}
            outs[_arm].append(rec)
            accs[_arm].append(acc)
            nlls[_arm].append(nll / max(1, int(tt.shape[0])))
            del lg, lp
        del ent
        if (_j + 1) % 250 == 0:
            print('lambada %d/1000' % (_j + 1), flush=True)
    return outs, {_a: sum(accs[_a]) / max(1, len(accs[_a])) for _a in ARMS}, {_a: sum(nlls[_a]) / max(1, len(nlls[_a])) for _a in ARMS}, accs, nlls
val_nll = _block_nll_all(val_full_chk)
print('full-val NLL: %s' % ({k: round(v, 5) for k, v in val_nll.items()}), flush=True)
_mark('full-val routed (1024 blocks x 5 passes)')
dom_nll = {}
for _d in ('general', 'code', 'math', 'scientific', 'multilingual'):
    _guard()
    dom_nll[_d] = _block_nll_all(dom_blocks[_d])
    print('domain %s NLL: %s' % (_d, {k: round(v, 5) for k, v in dom_nll[_d].items()}), flush=True)
_mark('domains routed (5 x 64 blocks x 5 passes)')
hs_outs, hs_acc, hs_corr = _hs_all()
print('HS acc: %s' % ({k: round(v, 4) for k, v in hs_acc.items()}), flush=True)
_mark('hellaswag routed (1000 x 4 opts x 5 passes)')
lam_outs, lam_acc, lam_nll, lam_corr, lam_n = _lam_all()
print('LAMBADA acc: %s nll: %s' % ({k: round(v, 4) for k, v in lam_acc.items()}, {k: round(v, 5) for k, v in lam_nll.items()}), flush=True)
_mark('lambada routed (1000 x 5 passes)')
for _arm in ARMS:
    (EOUT / f'hellaswag-{_arm}.jsonl').write_text('\n'.join(json.dumps(x) for x in hs_outs[_arm]))
    (EOUT / f'lambada-{_arm}.jsonl').write_text('\n'.join(json.dumps(x) for x in lam_outs[_arm]))
budget = {m: s / max(1, n) for m, (s, n) in rt.applied.items()}
_bmeans = [budget[m] for m in ARMS]
assert max(_bmeans) - min(_bmeans) < 1e-6 * max(1e-9, max(_bmeans)), 'per-arm alpha8 budget diverged'
print('budget: per-arm mean applied a8=%s (multisets exact)' % ({k: round(v, 6) for k, v in budget.items()}), flush=True)
inv = {
    'learned_val_nll_vs_B': float(val_nll['learned'] - B_REF['val_nll']),
    'learned_hs_acc_vs_B': float(hs_acc['learned'] - B_REF['hs_acc']),
    'learned_lam_acc_vs_B': float(lam_acc['learned'] - B_REF['lam_acc']),
    'learned_lam_nll_vs_B': float(lam_nll['learned'] - B_REF['lam_nll']),
}
assert abs(inv['learned_val_nll_vs_B']) < 5e-3, 'learned arm does not reproduce arb-B val NLL'
assert abs(inv['learned_hs_acc_vs_B']) <= 0.003, 'learned arm does not reproduce arb-B HS acc'
assert abs(inv['learned_lam_nll_vs_B']) < 5e-3, 'learned arm does not reproduce arb-B LAMBADA NLL'
print('INVARIANT ok: learned reproduces arb-B (dval=%.2e dhs=%.4f dlam-nll=%.2e)' % (inv['learned_val_nll_vs_B'], inv['learned_hs_acc_vs_B'], inv['learned_lam_nll_vs_B']), flush=True)
boot = {}
for _arm in ('shuffled', 'hard', 'easy'):
    boot[f'hs-{_arm}-vs-learned'] = _boot_diff(hs_corr[_arm], hs_corr['learned'])
    boot[f'lam-acc-{_arm}-vs-learned'] = _boot_diff(lam_corr[_arm], lam_corr['learned'])
    boot[f'lam-nll-{_arm}-vs-learned'] = _boot_diff(lam_n[_arm], lam_n['learned'])
    print('HS %s-vs-learned: dacc %+.4f p=%.4g | LAM-nll: dnll %+.5f p=%.4g' % (_arm, boot[f'hs-{_arm}-vs-learned']['mean'], boot[f'hs-{_arm}-vs-learned']['p'], boot[f'lam-nll-{_arm}-vs-learned']['mean'], boot[f'lam-nll-{_arm}-vs-learned']['p']), flush=True)
(EOUT / 'bootstrap.json').write_text(json.dumps(boot, indent=2))
(EOUT / 'routing.json').write_text(json.dumps({'arms': list(ARMS), 'val_nll': val_nll, 'domains': dom_nll, 'hs_acc': hs_acc, 'lam_acc': lam_acc, 'lam_nll': lam_nll, 'budget': budget, 'invariant_vs_arbB': inv, 'note': 'per-sequence alpha8 multiset exact in all arms; only placement differs; IDX2 fixed 1.25'}, indent=2))
(EOUT / 'config-route.json').write_text(json.dumps({'seeds': {'eval': C.EVAL_SEED, 'ple': C.SEED, 'shuffle': 1234}, 'arms': list(ARMS), 'frozen_ref': REF, 'arb_B_ref': B_REF, 'note': 'inference-only routing diagnostic; backbone/PLE/reader/arb frozen; no R4'}, indent=2))
(EOUT / 'timings-route.json').write_text(json.dumps(timings, indent=2))
rt.close()
inj.close()
_mark('STAGE A2-ARB-ROUTE COMPLETE: routing is causal probe; no training; no R4')
