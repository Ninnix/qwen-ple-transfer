# Stage A2-arb-diag: analysis-only alpha8 gate diagnostic on frozen REAL-5M R=1 (IDX 2+8).
# No training: backbone, PLE, reader, arb weights all frozen; inference only; no R4.
# Question: does IDX8 open for easy/local/predictable continuation and close for hard/long-range ones?
# Primary: alpha8 (arm B deployment) vs disabled-baseline next-token entropy/confidence.
# Buckets: position, frequency, punct/ws/newline, code keyword/ident, address reuse + trigram repeat.
# HellaSwag excluded (hellaswag skipped: ranking task has no per-position prediction targets; LAMBADA covers long-range).
# Gate index t pairs with prediction of ids[t+1]; target-token features use ids[t+1].
# Persists: alpha8-diag.json, config-diag.json, timings-diag.json. No weight writes.
import hashlib, json, math, re, string, time
import torch, torch.nn.functional as F
from pathlib import Path
from torch import nn
T0 = time.perf_counter()
EOUT = Path('/kaggle/working/eval-stageA2-arb-diag'); EOUT.mkdir(parents=True, exist_ok=True)
EFI_MOUNT = Path('/kaggle/input/qwen-ple-reader-checkpoints')
ARB_MOUNT = Path('/kaggle/input/qwen-ple-a2-arb')
RDIR = Path('/kaggle/input/qwen-ple-a2-v34')
DLDIR = Path('/kaggle/working/ckpt-dl-arb-diag'); DLDIR.mkdir(parents=True, exist_ok=True)
assert RDIR.exists(), 'results dataset missing: attach ninnix/qwen-ple-a2-v34'
assert ARB_MOUNT.exists(), 'arb dataset missing: upload arb outputs (arbitration-b/c.pt+json, alpha8-stats.json) as a dataset and attach it'
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
VAL_SUB_N = 128  # 128 x 512 = 65536 tokens of frozen full-val
DOM_SUB_N = 32   # 32 x 512 per domain
B_INIT = -1.0986122886681098
assert trev == REF['target_rev'], 'tokenizer revision drift: %s' % trev
assert srev.startswith(REF['source_rev_prefix']), 'source revision drift'
print('frozen revisions ok (tokenizer/model/ple)', flush=True)
print('diag: analysis-only, B-deployment gates vs disabled baseline; no R4, no weight updates', flush=True)
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
print('arb outputs pinned (B/C weights + stats SHAs match)', flush=True)
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
val_full_chk, mfull_chk = load_validation('full')
assert mfull_chk['tokens_sha256'] == REF['val_full_sha256'], 'frozen full-val drift'
dom_blocks = {}
for _d in ('general', 'code', 'math', 'scientific', 'multilingual'):
    _b, _m = build_domain_blocks(tokenizer, _d, C.EVAL_DOMAIN_TOKENS)
    assert _m['tokens_sha256'] == REF['domains'][_d], 'domain token drift: ' + _d
    dom_blocks[_d] = _b
print('frozen full-val + 5 domains ok', flush=True)
lam_rows, lam_meta = _lambada_rows(C.EVAL_MC_N)
assert lam_meta['dataset'] == REF['mc']['lambada']['dataset'] and str(lam_meta['rev']) == REF['mc']['lambada']['rev'], 'lambada drift'
assert len(lam_rows) == REF['mc']['lambada']['n'], 'lambada count drift'
print('lambada manifest ok (n=1000)', flush=True)
_mark('frozen inputs verified')
VAL_SUB = val_full_chk[:VAL_SUB_N]
DOM_SUB = {d: dom_blocks[d][:DOM_SUB_N] for d in dom_blocks}
assert VAL_SUB.shape == (128, 512), 'val subset shape drift'
assert all(v.shape == (32, 512) for v in DOM_SUB.values()), 'domain subset shape drift'
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
    # Read-only copy of the arb gate math. Feature uses h8.detach(): no graph.
    def __init__(self, w, b):
        super().__init__()
        self.w = w.float()
        self.b = float(b)
    def alpha8(self, h8):
        hn = rms_norm(h8.detach().float())
        return 0.5 * torch.sigmoid((hn * self.w.to(hn.device)).sum(-1) + self.b)
stB = torch.load(str(ARB_MOUNT / 'arbitration-b.pt'), map_location='cpu', weights_only=True)
stC = torch.load(str(ARB_MOUNT / 'arbitration-c.pt'), map_location='cpu', weights_only=True)
assert tuple(stB['w'].shape) == (1024,) and tuple(stC['w'].shape) == (1024,), 'arb w shape drift'
gB = GateOnly(stB['w'], stB['b'])
gC = GateOnly(stC['w'], stC['b'])
assert abs(float(stB['b']) - float(json.loads((ARB_MOUNT / 'arbitration-b.json').read_text())['b'])) < 1e-9, 'pt/json b drift'
print('arb B/C weights loaded (frozen, inference only)', flush=True)
class ArbCapture(nn.Module):
    # Deployed-B reproduction: fixed IDX2=1.25 plus dynamic IDX8 injection; records a8.
    def __init__(self, model, frozen, gate):
        super().__init__()
        self.gate = gate
        self.fr2 = frozen.readers['2']
        self.fr8 = frozen.readers['8']
        self.memory = None
        self.last_alpha8 = None
        dec = decoder_layers(model)
        self.handles = [
            dec[2].register_forward_pre_hook(self._hook2(), with_kwargs=True),
            dec[8].register_forward_pre_hook(self._hook8(), with_kwargs=True),
        ]
    def _hook2(self):
        def fn(mod, args, kw):
            if self.memory is None:
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
            if self.memory is None:
                return args, kw
            h = args[0] if args else kw['hidden_states']
            m = self.memory if self.memory.shape[1] == h.shape[1] else self.memory[:, :h.shape[1]]
            a8 = self.gate.alpha8(h)
            self.last_alpha8 = a8.detach()
            aug = self.fr8(h, m)
            out = h + a8.to(h.dtype).unsqueeze(-1) * (aug - h)
            if args:
                return (out, *args[1:]), kw
            kw['hidden_states'] = out
            return args, kw
        return fn
    def set_memory(self, m):
        self.memory = m
    def close(self):
        [h.remove() for h in self.handles]; self.handles.clear()
def _build_compact_diag(pdir, seqs):
    import hashlib as _hl
    from array import array as _array
    from safetensors import safe_open as _so
    _real = []
    with torch.inference_mode():
        for _s in seqs:
            _real.append(ngram_indices(_s.unsqueeze(0).cpu()).reshape(-1))
    uniq = torch.unique(torch.cat(_real)).long().cpu()
    print('diag union addresses: %d' % len(uniq), flush=True)
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
    (pdir / 'compact.json').write_text(json.dumps({'format': 'qwen-ple-compact-a2-arb-diag', 'version': 1, 'address_count': _n, 'uniq_sha256': _uh, 'row_dim': C.ROW_DIM, 'scale': float(_ms.scale), 'ple_revision': _ms.ple_revision, 'addrs_sha256': _ha.hexdigest(), 'rows_sha256': _hr.hexdigest()}, indent=2))
    _cc = CompactPLE(pdir)
    _g = torch.Generator().manual_seed(0)
    _samp = _uq[torch.randint(0, len(_uq), (2048,), generator=_g)].reshape(128, 16)
    assert (_cc.lookup(_samp) - _ms.lookup(_samp)).abs().max().item() == 0.0
    return _cc
_seq_all = [VAL_SUB[i] for i in range(VAL_SUB.shape[0])]
for _d in DOM_SUB:
    for _bi in range(DOM_SUB[_d].shape[0]):
        _seq_all.append(DOM_SUB[_d][_bi])
_lam_ids = []
for r in lam_rows:
    _p = tokenizer(r['context'], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    _t = tokenizer((' ' + r['target'].strip() if not r['target'].startswith(' ') else r['target']), return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    _lam_ids.append(torch.cat([_p, _t], dim=0)[:512])
    _seq_all.append(_lam_ids[-1])
_cgd = _build_compact_diag(WORK / 'compact-a2-arb-diag', _seq_all)
_mark('compact ready (diag subset union)')
cap = ArbCapture(model, inj, gB)
def _disabled_logits(ids):
    inj.set_memory(None)
    cap.set_memory(None)
    with torch.inference_mode():
        return model(input_ids=ids.to('cuda'), use_cache=False).logits
def _gate_alpha8(ids):
    with torch.inference_mode():
        cap.set_memory(_cgd.lookup(addresses(ids.cpu())).to('cuda'))
        _ = model(input_ids=ids.to('cuda'), use_cache=False).logits
        return cap.last_alpha8.float().cpu(), ids[0].cpu()
_probe = torch.cat([VAL_SUB[:4], DOM_SUB['general'][:2], DOM_SUB['code'][:1]], dim=0)
assert _probe.shape == (7, 512), 'probe shape drift'
_lg1 = _disabled_logits(_probe)
_lg2 = _disabled_logits(_probe)
assert torch.equal(_lg1.cpu(), _lg2.cpu()), 'eval path not deterministic'
print('eval determinism ok', flush=True)
del _lg1, _lg2
with torch.inference_mode():
    cap.set_memory(_cgd.lookup(addresses(_probe.cpu())).to('cuda'))
    _ = model(input_ids=_probe.to('cuda'), use_cache=False).logits
    _a8B = cap.last_alpha8.float().reshape(-1)
cap.gate = gC
with torch.inference_mode():
    cap.set_memory(_cgd.lookup(addresses(_probe.cpu())).to('cuda'))
    _ = model(input_ids=_probe.to('cuda'), use_cache=False).logits
    _a8C = cap.last_alpha8.float().reshape(-1)
cap.gate = gB
_bc = (_a8B - _a8C).abs()
print('B-vs-C probe: max|da8|=%.2e mean|da8|=%.2e' % (float(_bc.max()), float(_bc.mean())), flush=True)
assert float(_bc.mean()) < 2e-3, 'B/C policies diverged; B-only reporting invalid'
print('B/C equivalence ok: reporting B deployment only', flush=True)
del _a8B, _a8C, _bc, _probe
_mark('probe ok (determinism + B/C equivalence)')
CHUNK = 128
def _base_stats(ids):
    # Disabled-baseline per-position stats for targets ids[1:]: entropy, maxprob, nll, top1-correct.
    lg = _disabled_logits(ids.unsqueeze(0)).float().cpu()[0]
    T = ids.shape[0]
    ents, mps, nlls, oks = [], [], [], []
    for _s in range(0, T - 1, CHUNK):
        _e = min(_s + CHUNK, T - 1)
        lp = torch.log_softmax(lg[_s:_e], dim=-1)
        tgt = ids[_s + 1:_e + 1]
        pr = lp.exp()
        ents.append((-pr * lp).sum(-1))
        mps.append(pr.max(-1).values)
        nlls.append(-lp.gather(1, tgt.unsqueeze(1)).squeeze(1))
        oks.append((lp.argmax(-1) == tgt).to(torch.float32))
        del lp, pr
    del lg
    return torch.cat(ents), torch.cat(mps), torch.cat(nlls), torch.cat(oks)
_b0 = VAL_SUB[0]
_e0, _m0, _n0, _o0 = _base_stats(_b0)
_a80, _ = _gate_alpha8(_b0.unsqueeze(0))
assert _e0.shape[0] == 511 and _a80.reshape(-1).shape[0] == 512, 'position count drift'
print('position pairing ok: 511 targets per 512-block', flush=True)
del _e0, _m0, _n0, _o0, _a80
def _block_positions(ids):
    # Returns dict of per-target-position feature streams for one 512-block.
    ent, mxp, nll, ok = _base_stats(ids)
    a8full, _ = _gate_alpha8(ids.unsqueeze(0))
    a8 = a8full.reshape(-1)[:511]
    assert ent.shape[0] == 511, 'target count drift'
    tgt = ids[1:].long()
    addrs = ngram_indices(ids.unsqueeze(0).cpu()).reshape(512, 16)
    reuse = torch.zeros(511, dtype=torch.bool)
    for t in range(511):
        _lo = max(0, t - 64)
        prev = set(addrs[_lo:t].reshape(-1).tolist()) if t > 0 else set()
        reuse[t] = bool(set(addrs[t].tolist()) & prev)
    tri = torch.zeros(511, dtype=torch.bool)
    tl = ids.tolist()
    for t in range(2, 511):
        tri[t] = tl[t - 2:t + 1] in [tl[j - 2:j + 1] for j in range(2, t)]
    return {'a8': a8, 'ent': ent, 'mxp': mxp, 'nll': nll, 'ok': ok, 'tgt': tgt, 'pos': torch.arange(511), 'reuse': reuse, 'tri': tri}
S = {'a8': [], 'ent': [], 'mxp': [], 'nll': [], 'ok': [], 'tgt': [], 'pos': [], 'reuse': [], 'tri': [], 'src': []}
SRC_ID = {'full-val': 0, 'general': 1, 'code': 2, 'math': 3, 'scientific': 4, 'multilingual': 5, 'lambada': 6}
def _acc_block(ids, src):
    _guard()
    d = _block_positions(ids)
    for k in ('a8', 'ent', 'mxp', 'nll', 'ok', 'tgt', 'pos'):
        S[k].append(d[k])
    S['reuse'].append(d['reuse'])
    S['tri'].append(d['tri'])
    S['src'].append(torch.full((511,), SRC_ID[src], dtype=torch.long))
    del d
for _bi in range(VAL_SUB.shape[0]):
    _acc_block(VAL_SUB[_bi], 'full-val')
    if (_bi + 1) % 32 == 0:
        print('val-sub %d/128' % (_bi + 1), flush=True)
for _d in DOM_SUB:
    for _bi in range(DOM_SUB[_d].shape[0]):
        _acc_block(DOM_SUB[_d][_bi], _d)
    print('domain %s done' % _d, flush=True)
_mark('block subsets scored (val-sub + 5 domains)')
def _lam_positions(p_ids, t_len):
    ent, mxp, nll, ok = _base_stats(p_ids)
    a8full, _ = _gate_alpha8(p_ids.unsqueeze(0))
    a8 = a8full.reshape(-1)[:p_ids.shape[0] - 1]
    assert ent.shape[0] == p_ids.shape[0] - 1, 'lambada target count drift'
    keep = torch.arange(p_ids.shape[0] - 1) >= (p_ids.shape[0] - 1 - t_len)
    return {'a8': a8[keep], 'ent': ent[keep], 'mxp': mxp[keep], 'nll': nll[keep], 'ok': ok[keep], 'tgt': p_ids[1:][keep].long()}
L = {'a8': [], 'ent': [], 'mxp': [], 'nll': [], 'ok': [], 'tgt': []}
for _j, _ids in enumerate(_lam_ids):
    _guard()
    _t = tokenizer((' ' + lam_rows[_j]['target'].strip() if not lam_rows[_j]['target'].startswith(' ') else lam_rows[_j]['target']), add_special_tokens=False)['input_ids']
    d = _lam_positions(_ids, len(_t))
    for k in L:
        L[k].append(d[k])
    if (_j + 1) % 250 == 0:
        print('lambada %d/1000' % (_j + 1), flush=True)
_mark('lambada scored (target spans only)')
for k in S:
    S[k] = torch.cat(S[k]) if S[k] and S[k][0].dtype != torch.bool else torch.cat(S[k]).to(torch.bool) if S[k] else torch.tensor([])
for k in L:
    L[k] = torch.cat(L[k])
A8 = torch.cat([S['a8'], L['a8']])
NTOT = int(A8.numel())
print('positions: blocks=%d lambada-toks=%d total=%d' % (int(S['a8'].numel() // 511), int(L['a8'].numel()), NTOT), flush=True)
def _pearson(x, y):
    xc = x - x.mean(); yc = y - y.mean()
    return float((xc * yc).mean() / (x.std(correction=0) * y.std(correction=0) + 1e-12))
def _rank(v):
    return torch.argsort(torch.argsort(v)).float()
def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))
def _ms(v):
    return {'mean': float(v.float().mean()), 'std': float(v.float().std(correction=0)), 'n': int(v.numel())}
def _deciles(x, y, k=10):
    qs = torch.quantile(x, torch.arange(1, k).float() / k)
    edges = torch.cat([torch.tensor([float('-inf')]), qs, torch.tensor([float('inf')])])
    out = []
    for i in range(k):
        m = (x >= edges[i]) & (x <= edges[i + 1] if i == k - 1 else x < edges[i + 1])
        out.append({'lo': float(edges[i]), 'hi': float(edges[i + 1]), **_ms(y[m])})
    return out
BA8, BE, BM, BN, BO = S['a8'], S['ent'], S['mxp'], S['nll'], S['ok']
LA8, LE, LM, LN, LO = L['a8'], L['ent'], L['mxp'], L['nll'], L['ok']
primary = {
    'blocks': {
        'a8_vs_entropy': {'pearson': _pearson(BA8, BE), 'spearman': _spearman(BA8, BE)},
        'a8_vs_maxprob': {'pearson': _pearson(BA8, BM), 'spearman': _spearman(BA8, BM)},
        'a8_vs_nll': {'pearson': _pearson(BA8, BN), 'spearman': _spearman(BA8, BN)},
        'a8_correct_top1': _ms(BA8[BO > 0.5]),
        'a8_wrong_top1': _ms(BA8[BO < 0.5]),
        'a8_easy_nll_lt1': _ms(BA8[BN < 1.0]),
        'a8_hard_nll_gt4': _ms(BA8[BN > 4.0]),
        'entropy_deciles': _deciles(BE, BA8),
        'maxprob_bins': _deciles(BM, BA8),
    },
    'lambada': {
        'a8_vs_entropy': {'pearson': _pearson(LA8, LE), 'spearman': _spearman(LA8, LE)},
        'a8_vs_maxprob': {'pearson': _pearson(LA8, LM), 'spearman': _spearman(LA8, LM)},
        'a8_vs_nll': {'pearson': _pearson(LA8, LN), 'spearman': _spearman(LA8, LN)},
        'a8_correct_top1': _ms(LA8[LO > 0.5]),
        'a8_wrong_top1': _ms(LA8[LO < 0.5]),
    },
}
POS_EDGES = [0, 16, 64, 256, 511]
pos_bins = []
for i in range(4):
    m = (S['pos'] >= POS_EDGES[i]) & (S['pos'] < POS_EDGES[i + 1])
    pos_bins.append({'lo': POS_EDGES[i], 'hi': POS_EDGES[i + 1], **_ms(S['a8'][m])})
freq = torch.bincount(torch.cat([S['tgt'], L['tgt']]), minlength=C.VOCAB)
order = torch.argsort(freq, descending=True)
rankof = torch.empty(C.VOCAB, dtype=torch.long); rankof[order] = torch.arange(C.VOCAB)
trank = rankof[torch.cat([S['tgt'], L['tgt']])]
AA = torch.cat([S['a8'], L['a8']])
freq_bins = []
for _lo, _hi, _nm in [(0, 100, 'top100'), (100, 1000, 'r100_1k'), (1000, 10000, 'r1k_10k'), (10000, C.VOCAB, 'tail')]:
    m = (trank >= _lo) & (trank < _hi)
    freq_bins.append({'name': _nm, **_ms(AA[m])})
PUNCT = set(string.punctuation)
def _charclass(s):
    if '\n' in s:
        return 'newline'
    t = s.strip()
    if t == '':
        return 'whitespace'
    if all(c in PUNCT for c in t):
        return 'punct'
    if t.replace('_', '').isalnum():
        return 'word'
    return 'mixed'
_tgts = torch.cat([S['tgt'], L['tgt']]).tolist()
_cc = [_charclass(tokenizer.decode([i])) for i in _tgts]
char_bins = []
for _nm in ('newline', 'whitespace', 'punct', 'word', 'mixed'):
    m = torch.tensor([c == _nm for c in _cc])
    char_bins.append({'name': _nm, **_ms(AA[m])})
PY_KW = set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case'.split())
def _codeclass(s):
    t = s.strip()
    if t in PY_KW:
        return 'keyword'
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', t or ' '):
        return 'ident'
    if t and t[0].isdigit():
        return 'num'
    if t == '' or '\n' in s:
        return 'space_nl'
    return 'other'
_codem = S['src'] == SRC_ID['code']
_codet = S['tgt'][_codem].tolist()
_cls = [_codeclass(tokenizer.decode([i])) for i in _codet]
code_bins = []
for _nm in ('keyword', 'ident', 'num', 'space_nl', 'other'):
    m = torch.tensor([c == _nm for c in _cls])
    code_bins.append({'name': _nm, **_ms(S['a8'][_codem][m])})
reuse_bins = {'reuse': _ms(S['a8'][S['reuse']]), 'novel': _ms(S['a8'][~S['reuse']]), 'trigram_repeat': _ms(S['a8'][S['tri']]), 'trigram_novel': _ms(S['a8'][~S['tri']])}
diag = {
    'arm': 'B-deployment (C-equivalent on probe)',
    'n_block_positions': int(S['a8'].numel()),
    'n_lambada_positions': int(L['a8'].numel()),
    'primary': primary,
    'position_bins': pos_bins,
    'freq_buckets': freq_bins,
    'char_classes': char_bins,
    'code_classes': code_bins,
    'reuse': reuse_bins,
}
(EOUT / 'alpha8-diag.json').write_text(json.dumps(diag, indent=2))
(EOUT / 'config-diag.json').write_text(json.dumps({'seeds': {'eval': C.EVAL_SEED, 'ple': C.SEED}, 'val_subset_blocks': VAL_SUB_N, 'domain_subset_blocks': DOM_SUB_N, 'lambada_rows': len(lam_rows), 'frozen_ref': REF, 'note': 'analysis-only; backbone/PLE/reader/arb frozen; no R4; hellaswag skipped (ranking task)'}, indent=2))
(EOUT / 'timings-diag.json').write_text(json.dumps(timings, indent=2))
cap.close()
inj.close()
_mark('STAGE A2-ARB-DIAG COMPLETE: entropy-first gate diagnostic; no training; no R4')
pb = primary['blocks']
print('PRIMARY blocks: r(a8,ent)=%.3f rho=%.3f | r(a8,maxp)=%.3f rho=%.3f | r(a8,nll)=%.3f rho=%.3f' % (pb['a8_vs_entropy']['pearson'], pb['a8_vs_entropy']['spearman'], pb['a8_vs_maxprob']['pearson'], pb['a8_vs_maxprob']['spearman'], pb['a8_vs_nll']['pearson'], pb['a8_vs_nll']['spearman']), flush=True)
print('EASY/HARD blocks: nll<1 mean=%.4f | nll>4 mean=%.4f | top1-ok=%.4f top1-bad=%.4f' % (pb['a8_easy_nll_lt1']['mean'], pb['a8_hard_nll_gt4']['mean'], pb['a8_correct_top1']['mean'], pb['a8_wrong_top1']['mean']), flush=True)
pl = primary['lambada']
print('PRIMARY lambada: r(a8,ent)=%.3f rho=%.3f | r(a8,nll)=%.3f rho=%.3f | ok=%.4f bad=%.4f' % (pl['a8_vs_entropy']['pearson'], pl['a8_vs_entropy']['spearman'], pl['a8_vs_nll']['pearson'], pl['a8_vs_nll']['spearman'], pl['a8_correct_top1']['mean'], pl['a8_wrong_top1']['mean']), flush=True)
for b in pos_bins:
    print('pos [%d,%d): mean=%.4f n=%d' % (b['lo'], b['hi'], b['mean'], b['n']), flush=True)
for b in freq_bins + char_bins + code_bins:
    print('%s: mean=%.4f n=%d' % (b['name'], b['mean'], b['n']), flush=True)
print('reuse=%.4f novel=%.4f | tri-rep=%.4f tri-new=%.4f' % (reuse_bins['reuse']['mean'], reuse_bins['novel']['mean'], reuse_bins['trigram_repeat']['mean'], reuse_bins['trigram_novel']['mean']), flush=True)
