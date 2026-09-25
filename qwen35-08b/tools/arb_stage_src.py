# Stage A2-arb: tiny dynamic memory-arbitration on frozen REAL-5M R=1 (IDX 2+8).
# Frozen: backbone, PLE, W_K, W_V, beta, trained gammas. Only alpha2_raw, w, b trainable (~1K params).
# IDX2: one bounded trainable global scalar alpha2, init 1.25, range 0.75-1.75.
# IDX8: token/context-dependent alpha8_t = 0.5 * sigmoid(w . RMSNorm(h8_t) + b), init ~0.125 constant.
# Effective injections: IDX2: gamma2 * alpha2 ; IDX8: gamma8 * alpha8_t (in-memory only, base weights never overwritten).
# Arms: A static canonical IDX2=1.25 IDX8=0.125 ; B dynamic IDX8 with IDX2=1.25 fixed ; C dynamic IDX8 + learned global IDX2 scalar.
# Train: small balanced calibration corpus ~319K tokens (general, narrative, code, math, scientific, multilingual). No FineWeb-only calibration.
# Light regularizer on the mean IDX8 multiplier toward ~0.125 (mean-only, per-token variation allowed). No R4.
# Gate feature uses h8.detach(): no backbone autograd graph for the gate. Pre-run gates before training:
# INVARIANT 1 (fresh A == persisted v40 calibrated, tolerance) and INVARIANT 2 (init-B/C == A, exact gates).
# Persists: hellaswag-B.jsonl lambada-B.jsonl hellaswag-C.jsonl lambada-C.jsonl arbitration-b.json arbitration-c.json arbitration-b.pt arbitration-c.pt alpha8-stats.json calibration.json
import hashlib, json, math, random, re, subprocess, sys, time, zlib
import torch, torch.nn.functional as F
from pathlib import Path
from torch import nn
T0 = time.perf_counter()
EOUT = Path('/kaggle/working/eval-stageA2-arb'); EOUT.mkdir(parents=True, exist_ok=True)
EFI_MOUNT = Path('/kaggle/input/qwen-ple-reader-checkpoints')
RDIR = Path('/kaggle/input/qwen-ple-a2-v34')
DLDIR = Path('/kaggle/working/ckpt-dl-arb'); DLDIR.mkdir(parents=True, exist_ok=True)
assert RDIR.exists(), 'results dataset missing: attach ninnix/qwen-ple-a2-v34'
REF = {
    'target_rev': '2fc06364715b967f1860aea9cf38778875588b17',
    'source_rev_prefix': '236dfdf28582',
    'ple_revision': '236dfdf285828023ca3bcd3f37366c58a3469b13',
    'ckpt_sha': {'real-5m': '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594'},
    'ckpt_tokens': {'real-5m': 5000192},
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
ARB_PER_DOMAIN = 53248  # 104 x 512 blocks per source
ARB_SOURCES = ('general', 'narrative', 'code', 'math', 'scientific', 'multilingual')
ARB_TOTAL = 319488  # 6 x 53248 = 624 x 512, within 250K-500K
ARB_SEED = 1234
ARB_LR = 1e-3
ARB_WD = 0.0
ARB_WARM = 32
ARB_LAMBDA = 2.0  # light regularizer weight on mean(alpha8_t)-0.125
ALPHA2_LO = 0.75
ALPHA2_SPAN = 1.0  # alpha2 in [0.75, 1.75]
ALPHA2_INIT = 1.25
ALPHA8_INIT = 0.125
B_INIT = -1.0986122886681098  # logit(0.25): 0.5*sigmoid(B_INIT)=0.125
STATIC_M2 = 1.25
STATIC_M8 = 0.125
assert ARB_TOTAL == ARB_PER_DOMAIN * 6 and 250000 <= ARB_TOTAL <= 500000, 'calibration budget must be 250K-500K'
assert abs(0.5 / (1 + math.exp(-B_INIT)) - 0.125) < 1e-12, 'B_INIT must give 0.125'
assert trev == REF['target_rev'], 'tokenizer revision drift: %s' % trev
assert srev.startswith(REF['source_rev_prefix']), 'source revision drift'
print('frozen revisions ok (tokenizer/model/ple)', flush=True)
print('arms: A static canonical IDX2=1.25 IDX8=0.125 | B dynamic IDX8 + IDX2=1.25 fixed | C dynamic IDX8 + learned global IDX2 scalar', flush=True)
print('calibration: 6-way balanced ~319K tokens (general, narrative, code, math, scientific, multilingual), no FineWeb-only calibration', flush=True)
timings = {}
def _mark(name):
    timings[name] = round(time.perf_counter() - T0, 1)
    print('[t=%ds] %s' % (timings[name], name), flush=True)
def _fetch_ckpt():
    need = ['protocol.json', 'real-5m-r1.reader.safetensors', 'real-5m-r1.run.json']
    if EFI_MOUNT.exists() and all((EFI_MOUNT / f).exists() for f in need):
        print('checkpoints: mounted dataset (schema-validated)', flush=True)
        return EFI_MOUNT
    _env = dict(os.environ); _env['KAGGLE_API_TOKEN'] = secret_value_1
    for _f in need:
        _r = subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', 'ninnix/qwen-ple-reader-checkpoints', '-f', _f, '-p', str(DLDIR), '--force'], capture_output=True, text=True, env=_env, timeout=1200)
        assert _r.returncode == 0 and (DLDIR / _f).exists(), 'checkpoint download failed: ' + _f
    print('checkpoints: api download (mount missing)', flush=True)
    return DLDIR
CKDIR = _fetch_ckpt()
_proto = json.loads((CKDIR / 'protocol.json').read_text())
assert _proto.get('ple_revision') == REF['ple_revision'], 'PLE revision drift'
def _file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
def _load_sfc(pt_name, run_name, expect_tokens, expect_sha):
    from safetensors.torch import load_file
    run = json.loads((CKDIR / run_name).read_text())
    assert run.get('token_count', run.get('tokens')) == expect_tokens, 'budget mismatch: ' + run_name
    assert (CKDIR / pt_name).exists(), 'missing: ' + pt_name
    h = _file_sha(CKDIR / pt_name)
    assert h == expect_sha, 'bytes mismatch: ' + pt_name
    return load_file(str(CKDIR / pt_name), device='cpu'), h
_cksha = {}
st5, _h5 = _load_sfc('real-5m-r1.reader.safetensors', 'real-5m-r1.run.json', REF['ckpt_tokens']['real-5m'], REF['ckpt_sha']['real-5m'])
_cksha['real-5m'] = _h5
print('arm ready: frozen REAL-5M R=1 weights (backbone, PLE, W_K, W_V, beta, gamma stay frozen)', flush=True)
_mark('checkpoints ready')
V = {}
for _f in ['config.json', 'summary.json', 'bootstrap.json', 'hellaswag.jsonl', 'lambada.jsonl']:
    assert (RDIR / _f).exists(), 'missing frozen input: ' + _f
V['config'] = json.loads((RDIR / 'config.json').read_text())
V['bootstrap'] = json.loads((RDIR / 'bootstrap.json').read_text())
V['rows'] = {}
for _t in ['hellaswag', 'lambada']:
    V['rows'][_t] = [json.loads(x) for x in (RDIR / f'{_t}.jsonl').read_text().splitlines() if x.strip()]
print('frozen v34 outputs loaded (HS+LAMBADA)', flush=True)
MCN = C.EVAL_MC_N
hs_rows, hs_meta = _hellaswag_rows(MCN)
lam_rows, lam_meta = _lambada_rows(MCN)
for _name, _rows, _meta in [('hellaswag', hs_rows, hs_meta), ('lambada', lam_rows, lam_meta)]:
    _ref = REF['mc'][_name]
    assert _meta['dataset'] == _ref['dataset'] and str(_meta['rev']) == _ref['rev'], 'dataset drift: ' + _name
    assert len(_rows) == _ref['n'] == len(V['rows'][_name]), 'row count drift: ' + _name
    for _a, _b in zip(_rows, V['rows'][_name]):
        if _name == 'lambada':
            assert _a['target'] == _b['target'], 'prompt drift: lambada idx %d' % _b['idx']
        else:
            assert _a['label'] == _b['label'], 'manifest drift: %s idx %d' % (_name, _b['idx'])
print('MC manifests identical to v34 (hellaswag+lambada)', flush=True)
val_full_chk, mfull_chk = load_validation('full')
assert mfull_chk['tokens_sha256'] == REF['val_full_sha256'], 'frozen full-val drift'
print('frozen full-val ok sha=%s n=%d' % (mfull_chk['tokens_sha256'][:16], val_full_chk.shape[0]), flush=True)
dom_blocks = {}
for _d in DOMAINS:
    _b, _m = build_domain_blocks(tokenizer, _d, C.EVAL_DOMAIN_TOKENS)
    assert _m['tokens_sha256'] == REF['domains'][_d], 'domain token drift: ' + _d
    assert _b.shape[0] == 64, 'domain block drift: ' + _d
    dom_blocks[_d] = _b
print('domain artifacts identical to v34 (5 SHAs match)', flush=True)
_mark('frozen inputs verified')
class MemoryArbitration(nn.Module):
    # Tiny gate on top of frozen reader. Only w, b, alpha2_raw trainable (~1K params: 1024+1+1=1026).
    # IDX2: alpha2 = 0.75 + 1.0*sigmoid(alpha2_raw), init 1.25, bounded ~0.75-1.75.
    # IDX8: alpha8_t = 0.5*sigmoid(w . RMSNorm(h8_t) + b), init ~0.125 constant (w=0, b=logit(0.25)).
    def __init__(self, hidden=1024):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(hidden))
        self.b = nn.Parameter(torch.tensor(float(B_INIT)))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0))
    def alpha2(self):
        return ALPHA2_LO + ALPHA2_SPAN * torch.sigmoid(self.alpha2_raw)
    def alpha8(self, h8):
        hn = rms_norm(h8.detach().float())  # feature only: detached so no backbone autograd graph is built for the gate
        logit = (hn * self.w.float()).sum(-1) + self.b.float()
        return 0.5 * torch.sigmoid(logit)
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
arb_probe = MemoryArbitration(hidden=C.HIDDEN)
assert arb_probe.param_count() == C.HIDDEN + 2 == 1026, 'arbitration must be ~1K params'
assert abs(float(arb_probe.alpha2().detach()) - 1.25) < 1e-6, 'alpha2 init must be 1.25'
_hp = torch.randn(2, 8, C.HIDDEN)
_a8i = arb_probe.alpha8(_hp).detach()
assert abs(float(_a8i.mean()) - 0.125) < 1e-6 and float(_a8i.std(correction=0)) < 1e-6, 'alpha8_t init must be ~constant 0.125'
assert float(arb_probe.alpha2().detach()) >= 0.75 and float(arb_probe.alpha2().detach()) <= 1.75, 'alpha2 range 0.75-1.75'
print('arbitration init ok: alpha2=1.25 in [0.75,1.75], alpha8_t~0.125 constant, params=%d (~1K)' % arb_probe.param_count(), flush=True)
del arb_probe, _hp, _a8i
inj = ReaderInjection(model, [2, 8], C.MEM_DIM, C.HIDDEN, 1, C.GAMMA_INIT).to('cuda')
def _load_arm_inmem(st):
    inj.load_state_dict({k: v.to('cuda') for k, v in st.items()})
    inj.requires_grad_(False)
    model.requires_grad_(False)
    assert sum(1 for p in inj.parameters() if p.requires_grad) == 0, 'inference-only: reader must stay grad-free'
    assert sum(1 for p in model.parameters() if p.requires_grad) == 0, 'backbone must stay frozen'
_load_arm_inmem(st5)
del st5
orig_g2 = float(inj.readers['2'].gamma.detach().cpu())
orig_g8 = float(inj.readers['8'].gamma.detach().cpu())
print('learned 5M gammas frozen: IDX2=%.6f IDX8=%.6f (W_K, W_V, beta, gamma never updated)' % (orig_g2, orig_g8), flush=True)
assert abs(orig_g2 - 0.06166713312268257) < 1e-6 and abs(orig_g8 - 0.07537073642015457) < 1e-6, '5M weights mismatch'
backbone_versions = {n: p._version for n, p in model.named_parameters()}
_mark('inj ready (frozen, grad-free)')
class ArbHooks(nn.Module):
    # Dynamic injection on top of frozen readers without modifying them.
    # IDX2: h = h + (gamma2 * alpha2) * o2 ; IDX8: h = h + (gamma8 * alpha8_t) * o8.
    # Frozen reader gives aug = h + gamma*o, so delta = alpha*(aug-h) is exact without touching gamma.
    # Only arb.w, arb.b, arb.alpha2_raw receive gradients; backbone/PLE/W_K/W_V/beta/gamma frozen.
    def __init__(self, model, frozen, arb):
        super().__init__()
        self.arb = arb
        self.fr2 = frozen.readers['2']
        self.fr8 = frozen.readers['8']
        self.fixed_alpha2 = None  # None = learned global IDX2 scalar; float = fixed (arm B uses 1.25)
        self.memory = None
        self.cur_alpha8 = None  # differentiable, for mean regularizer
        self.last_alpha8 = None  # detached, for stats
        dec = decoder_layers(model)
        self.handles = [
            dec[2].register_forward_pre_hook(self._hook2(), with_kwargs=True),
            dec[8].register_forward_pre_hook(self._hook8(), with_kwargs=True),
        ]
    def _a2(self):
        if self.fixed_alpha2 is not None:
            return torch.tensor(float(self.fixed_alpha2), device=self.arb.alpha2_raw.device)
        return self.arb.alpha2()
    def _hook2(self):
        def fn(mod, args, kw):
            assert self.memory is not None, 'arb memory missing at IDX2'
            h = args[0] if args else kw['hidden_states']
            m = self.memory
            if m.shape[1] != h.shape[1]:
                m = m[:, :h.shape[1]]
            aug = self.fr2(h, m)  # frozen: h + gamma2*o2
            a2 = self._a2().to(h.dtype)
            out = h + a2 * (aug - h)  # h + (gamma2 * alpha2) * o2
            if args:
                return (out, *args[1:]), kw
            kw['hidden_states'] = out
            return args, kw
        return fn
    def _hook8(self):
        def fn(mod, args, kw):
            assert self.memory is not None, 'arb memory missing at IDX8'
            h = args[0] if args else kw['hidden_states']
            m = self.memory
            if m.shape[1] != h.shape[1]:
                m = m[:, :h.shape[1]]
            a8 = self.arb.alpha8(h)  # 0.5*sigmoid(w.RMSNorm(h8_t)+b), [B,T]
            self.cur_alpha8 = a8  # keep graph for regularizer
            self.last_alpha8 = a8.detach()
            aug = self.fr8(h, m)  # frozen: h + gamma8*o8
            w8 = a8.to(h.dtype).unsqueeze(-1)
            out = h + w8 * (aug - h)  # h + (gamma8 * alpha8_t) * o8
            if args:
                return (out, *args[1:]), kw
            kw['hidden_states'] = out
            return args, kw
        return fn
    def set_memory(self, m):
        self.memory = m
    def close(self):
        [h.remove() for h in self.handles]; self.handles.clear()
def _arb_sources():
    # Balanced 6-way calibration: general, narrative, code, math, scientific, multilingual. No FineWeb-only calibration.
    # Code + multilingual mirror the frozen harness candidate order (script/gated-safe first) with fallback.
    return [
        {'key': 'general', 'ds': 'wikitext', 'cfg': 'wikitext-103-raw-v1', 'split': 'train', 'kind': 'text'},
        {'key': 'narrative', 'ds': 'roneneldan/TinyStories', 'cfg': None, 'split': 'train', 'kind': 'text'},
        {'key': 'code', 'ds': 'edward-io/starcoderdata-repo', 'cfg': None, 'split': 'train', 'kind': 'code', 'cands': [['edward-io/starcoderdata-repo', None], ['codeparrot/github-code-clean', None], ['bigcode/the-stack-smol', None]]},
        {'key': 'math', 'ds': 'openai/gsm8k', 'cfg': 'main', 'split': 'test', 'kind': 'qa'},
        {'key': 'scientific', 'ds': 'allenai/sciq', 'cfg': None, 'split': 'test', 'kind': 'sci'},
        {'key': 'multilingual', 'ds': 'wikipedia', 'cfg': '20220301.de', 'split': 'train', 'kind': 'text', 'cands': [['wikipedia', '20220301.de'], ['HuggingFaceFW/fineweb-2', 'deu_Latn']]},
    ]
def _calib_text(kind, row):
    if kind == 'text':
        return row.get('text') or ''
    if kind == 'code':
        return row.get('content') or row.get('code') or row.get('text') or ''
    if kind == 'qa':
        return (row.get('question') or '') + '\n' + (row.get('answer') or '')
    if kind == 'sci':
        return (row.get('support') or '') + '\n' + (row.get('question') or '') + '\n' + (row.get('correct_answer') or row.get('answer') or '')
    return ''
def build_calibration(tok):
    from array import array as _array
    from datasets import load_dataset as _ld
    from huggingface_hub import HfApi as _Api
    assert 'FineWebEduStream' not in dir(), 'calibration must not use FineWeb-only stream'
    per = {}
    metas = {}
    for src in _arb_sources():
        assert src['key'] in ARB_SOURCES, 'source key drift'
        raw = src.get('cands') or [[src['ds'], src['cfg']]]
        cands = [(c[0], c[1]) if isinstance(c, list) else (c, src['cfg']) for c in raw]
        assert all('fineweb-edu' not in c.lower() for c, _ in cands), 'calibration source must not be FineWeb-only'
        errs = []
        for _cand, _ccfg in cands:
            try:
                api = _Api(token=secret_value_0)
                drev = api.dataset_info(_cand).sha
                kw = {'revision': drev} if drev else {}
                if _ccfg is None:
                    ds = _ld(_cand, split=src['split'], streaming=True, **kw)
                else:
                    ds = _ld(_cand, _ccfg, split=src['split'], streaming=True, **kw)
                arr = _array('I')
                skip_arr = _array('I')
                docs = 0
                skipped_docs = 0
                for row in ds:
                    docs += 1
                    t = _calib_text(src['kind'], row)
                    if not t or not t.strip():
                        continue
                    ids = tok.encode(t, add_special_tokens=False) + [C.EOS]
                    if len(skip_arr) < C.EVAL_DOMAIN_TOKENS:
                        skip_arr.extend(ids)
                        skipped_docs += 1
                        if len(skip_arr) >= C.EVAL_DOMAIN_TOKENS:
                            del skip_arr[C.EVAL_DOMAIN_TOKENS:]
                        continue
                    arr.extend(ids)
                    if len(arr) >= ARB_PER_DOMAIN:
                        del arr[ARB_PER_DOMAIN:]
                        break
                assert len(arr) == ARB_PER_DOMAIN, (src['key'], len(arr))
                import hashlib as _hl
                digest = _hl.sha256(arr.tobytes()).hexdigest()
                per[src['key']] = torch.tensor(arr, dtype=torch.long).view(-1, 512)
                metas[src['key']] = {'dataset': _cand, 'config': _ccfg, 'split': src['split'], 'kind': src['kind'], 'dataset_rev': drev, 'docs': docs, 'skipped_prefix_docs': skipped_docs, 'tokens': len(arr), 'tokens_sha256': digest}
                print('calib %s: %s/%s sha=%s docs=%d (held-out prefix skipped)' % (src['key'], _cand, _ccfg, digest[:16], docs), flush=True)
                break
            except Exception as e:
                errs.append('%s: %s' % (_cand, str(e)[:140]))
                continue
        else:
            raise RuntimeError('no calibration candidate resolved for ' + src['key'] + ': ' + ' | '.join(errs))
    blocks = torch.cat([per[k] for k in ARB_SOURCES], dim=0)
    assert blocks.shape == (624, 512) and blocks.numel() == ARB_TOTAL, 'balanced 6x104 blocks = 319488 tokens'
    g = torch.Generator().manual_seed(ARB_SEED)
    perm = torch.randperm(blocks.shape[0], generator=g)
    blocks = blocks[perm]
    (EOUT / 'calibration.json').write_text(json.dumps({'total_tokens': ARB_TOTAL, 'per_domain_tokens': ARB_PER_DOMAIN, 'blocks': 624, 'seq': 512, 'seed': ARB_SEED, 'sources': metas, 'note': 'balanced general/narrative/code/math/scientific/multilingual; held-out 32K prefix skipped per overlapping source; no FineWeb-only calibration'}, indent=2))
    print('balanced calibration ready: %d tokens in %d blocks (6 x %d)' % (ARB_TOTAL, blocks.shape[0], ARB_PER_DOMAIN), flush=True)
    return blocks, metas
calib_blocks, calib_metas = build_calibration(tokenizer)
_mark('calibration built')
def _build_compact_arb(pdir, calib, uniq_extra=None):
    import hashlib as _hl
    from array import array as _array
    from safetensors import safe_open as _so
    _seqs = [calib[_i] for _i in range(calib.shape[0])]
    for _bi in range(val_full_chk.shape[0]):
        _seqs.append(val_full_chk[_bi])
    for _d in dom_blocks:
        for _bi in range(dom_blocks[_d].shape[0]):
            _seqs.append(dom_blocks[_d][_bi])
    for r in hs_rows:
        p = tokenizer(r['ctx'], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        for e in r['endings']:
            _seqs.append(torch.cat([p, tokenizer((' ' + e.strip() if not e.startswith(' ') else e), return_tensors='pt', add_special_tokens=False)['input_ids'][0]])[:512])
    for r in lam_rows:
        _seqs.append(torch.cat([tokenizer(r['context'], return_tensors='pt', add_special_tokens=False)['input_ids'][0], tokenizer((' ' + r['target'].strip() if not r['target'].startswith(' ') else r['target']), return_tensors='pt', add_special_tokens=False)['input_ids'][0]])[:512])
    _real = []
    with torch.inference_mode():
        for _s in _seqs:
            _real.append(ngram_indices(_s.unsqueeze(0).cpu()).reshape(-1))
    uniq = torch.unique(torch.cat(_real))
    print('arb union addresses: %d (calib+HS+LAMBADA+full-val+5domains)' % len(uniq), flush=True)
    del _real, _seqs
    uniq = uniq.long().cpu()
    _uh = _hl.sha256(_array('I', uniq.tolist()).tobytes()).hexdigest()
    need = True
    if (pdir / 'compact.json').exists():
        try:
            pm = json.loads((pdir / 'compact.json').read_text())
            need = not (pm.get('address_count') == len(uniq) and pm.get('uniq_sha256') == _uh and pm.get('ple_revision') == REF['ple_revision'])
        except Exception:
            need = True
    if need:
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
        (pdir / 'compact.json').write_text(json.dumps({'format': 'qwen-ple-compact-a2-arb', 'version': 1, 'address_count': _n, 'uniq_sha256': _uh, 'row_dim': C.ROW_DIM, 'scale': float(_ms.scale), 'ple_revision': _ms.ple_revision, 'addrs_sha256': _ha.hexdigest(), 'rows_sha256': _hr.hexdigest()}, indent=2))
        _cc = CompactPLE(pdir)
        _g = torch.Generator().manual_seed(0)
        _samp = _uq[torch.randint(0, len(_uq), (2048,), generator=_g)].reshape(128, 16)
        assert (_cc.lookup(_samp) - _ms.lookup(_samp)).abs().max().item() == 0.0
    return CompactPLE(pdir)
_cg = _build_compact_arb(WORK / 'compact-a2-arb', calib_blocks)
_mark('compact ready (no /kaggle/input random reads during training)')
def _alpha_stats(t):
    g = t.detach().float().reshape(-1)
    q = torch.quantile(g, torch.tensor([0.05, 0.5, 0.95]))
    return {'mean': float(g.mean()), 'std': float(g.std(correction=0)), 'p05': float(q[0]), 'p50': float(q[1]), 'p95': float(q[2]), 'n': int(g.numel())}
def train_arb(tag, learn_alpha2):
    # Tiny training: only arb.w, arb.b (+ arb.alpha2_raw for arm C) are trainable. Backbone, PLE, W_K, W_V, beta, gamma frozen.
    arb = MemoryArbitration(hidden=C.HIDDEN).to('cuda')
    assert arb.param_count() == 1026
    hooks = ArbHooks(model, inj, arb)
    inj.set_memory(None)
    for p in list(model.parameters()) + list(inj.parameters()):
        assert not p.requires_grad, 'backbone/reader must stay frozen during arbitration training'
    if not learn_alpha2:
        arb.alpha2_raw.requires_grad_(False)
        hooks.fixed_alpha2 = 1.25
    tr = [p for p in arb.parameters() if p.requires_grad]
    assert len(tr) == (3 if learn_alpha2 else 2), 'only w,b (+alpha2_raw for C) trainable'
    n_params = sum(p.numel() for p in tr)
    assert n_params in (1025, 1026), 'tiny ~1K params, got %d' % n_params
    print('train %s: learn_alpha2=%s params=%d names=%s' % (tag, learn_alpha2, n_params, ['w', 'b'] + (['alpha2_raw'] if learn_alpha2 else [])), flush=True)
    opt = torch.optim.AdamW(tr, lr=ARB_LR, weight_decay=ARB_WD)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in tr}, 'optimizer must contain exactly arbitration params'
    n_steps = calib_blocks.shape[0]
    def _lr(step):
        if step < ARB_WARM:
            return ARB_LR * (step + 1) / ARB_WARM
        t = min((step - ARB_WARM) / max(1, n_steps - ARB_WARM), 1.0)
        return ARB_LR * 0.5 * (1 + math.cos(math.pi * t))
    for g in opt.param_groups:
        g['lr'] = _lr(-1)
    tot_ce = 0.0; tot_reg = 0.0; tot_n = 0
    means = []
    model.train(False)
    for step in range(n_steps):
        if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
            print('HARD-GUARD: arbitration training early stop at step %d' % step, flush=True)
            break
        ids = calib_blocks[step].unsqueeze(0)
        mem = _cg.lookup(addresses(ids)).to('cuda')
        hooks.set_memory(mem)
        for g in opt.param_groups:
            g['lr'] = _lr(step)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=ids.to('cuda'), use_cache=False).logits
        ce = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), ids.to('cuda')[:, 1:].reshape(-1))
        a8 = hooks.cur_alpha8
        assert a8 is not None, 'alpha8 hook did not fire'
        mean_a8 = a8.float().mean()
        reg = ARB_LAMBDA * (mean_a8 - 0.125) ** 2  # light regularizer on the mean IDX8 multiplier toward ~0.125; mean-only, per-token variation allowed
        loss = ce + reg
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite arbitration loss at step %d' % step)
        loss.backward()
        assert not any(p.grad is not None for p in model.parameters()), 'backbone gradient leak'
        assert not any(p.grad is not None for p in inj.parameters()), 'frozen reader gradient leak (W_K, W_V, beta, gamma)'
        gn = torch.nn.utils.clip_grad_norm_(arb.parameters(), 1.0).item()
        opt.step()
        tot_ce += float(ce.detach()) * 511; tot_reg += float(reg.detach()) * 511; tot_n += 511
        means.append(float(mean_a8.detach()))
        if (step + 1) % 100 == 0:
            print('arb %s step %d/%d ce=%.4f reg=%.6f mean_a8=%.4f a2=%.4f lr=%.2e gnorm=%.3f' % (tag, step + 1, n_steps, tot_ce / tot_n, tot_reg / tot_n, float(mean_a8.detach()), float(arb.alpha2().detach()) if learn_alpha2 else 1.25, opt.param_groups[0]['lr'], gn), flush=True)
        del lg, ce, reg, loss, mem
    a2_final = float(arb.alpha2().detach()) if learn_alpha2 else 1.25
    assert 0.75 <= a2_final <= 1.75, 'learned alpha2 out of range'
    w_norm = float(arb.w.detach().float().norm())
    b_final = float(arb.b.detach())
    print('arb %s done: alpha2=%.4f w_norm=%.4f b=%.4f mean_a8_traj=%.4f' % (tag, a2_final, w_norm, b_final, sum(means) / max(1, len(means))), flush=True)
    torch.save({'w': arb.w.detach().cpu(), 'b': arb.b.detach().cpu(), 'alpha2_raw': arb.alpha2_raw.detach().cpu(), 'learn_alpha2': learn_alpha2, 'alpha2': a2_final, 'lr': ARB_LR, 'lambda': ARB_LAMBDA, 'steps': n_steps}, str(EOUT / f'arbitration-{tag}.pt'))
    (EOUT / f'arbitration-{tag}.json').write_text(json.dumps({'tag': tag, 'learn_alpha2': learn_alpha2, 'alpha2': a2_final, 'w_norm': w_norm, 'b': b_final, 'params': n_params, 'mean_alpha8_train': sum(means) / max(1, len(means)), 'lr': ARB_LR, 'wd': ARB_WD, 'lambda_mean_reg': ARB_LAMBDA, 'tokens': ARB_TOTAL, 'note': 'only alpha2/w/b trainable; backbone/PLE/W_K/W_V/beta/gamma frozen; mean-only reg toward 0.125'}, indent=2))
    hooks.close()
    cur = [n for n, p in model.named_parameters() if p._version != backbone_versions[n]]
    assert not cur, 'backbone weights changed during arbitration'
    return {'arb_state': {'w': arb.w.detach().cpu(), 'b': arb.b.detach().cpu(), 'alpha2_raw': arb.alpha2_raw.detach().cpu()}, 'alpha2': a2_final, 'learn_alpha2': learn_alpha2}
tokenizer.padding_side = 'left'
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = C.EOS
def _set_gamma_pair(m2, m8):
    with torch.no_grad():
        inj.readers['2'].gamma.copy_(torch.tensor(float(orig_g2 * m2)))
        inj.readers['8'].gamma.copy_(torch.tensor(float(orig_g8 * m8)))
def _restore_learned():
    with torch.no_grad():
        inj.readers['2'].gamma.copy_(torch.tensor(float(orig_g2)))
        inj.readers['8'].gamma.copy_(torch.tensor(float(orig_g8)))
def _logprob_opts(prompt, conts):
    p = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
    outs = []
    with torch.inference_mode():
        for c in conts:
            t = tokenizer(c if c.startswith(' ') else ' ' + c, return_tensors='pt', add_special_tokens=False)['input_ids']
            ids = torch.cat([p, t], dim=1)
            inj.set_memory(_cg.lookup(addresses(ids)).to('cuda'))
            lg = model(input_ids=ids.to('cuda'), use_cache=False).logits.float()
            lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
            outs.append(lp.gather(1, t[0].to('cuda').unsqueeze(1)).sum().item())
    return outs
def _lambada_score(context, target):
    p = tokenizer(context, return_tensors='pt', add_special_tokens=False)['input_ids']
    t = tokenizer(target if target.startswith(' ') else ' ' + target, return_tensors='pt', add_special_tokens=False)['input_ids']
    ids = torch.cat([p, t], dim=1)
    with torch.inference_mode():
        inj.set_memory(_cg.lookup(addresses(ids)).to('cuda'))
        lg = model(input_ids=ids.to('cuda'), use_cache=False).logits.float()
        lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
        nll = -lp.gather(1, t[0].to('cuda').unsqueeze(1)).sum().item()
        acc = int(bool((lp.argmax(-1).cpu() == t[0]).all()))
    return nll, acc, int(t.shape[1])
def _block_nll(blocks):
    out = []
    with torch.inference_mode():
        for bi in range(blocks.shape[0]):
            if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
                print('HARD-GUARD: block NLL early stop', flush=True)
                break
            b = blocks[bi].unsqueeze(0)
            inj.set_memory(_cg.lookup(addresses(b.cpu())).to('cuda'))
            lg = model(input_ids=b.to('cuda'), use_cache=False).logits
            s = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), b.to('cuda')[:, 1:].reshape(-1), reduction='sum').item()
            out.append({'sum': s, 'n': int(b[:, 1:].numel())})
    mean = sum(x['sum'] for x in out) / max(1, sum(x['n'] for x in out))
    return out, mean
def _block_nll_disabled(blocks):
    out = []
    with torch.inference_mode():
        inj.set_memory(None)
        for bi in range(blocks.shape[0]):
            if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
                print('HARD-GUARD: disabled block NLL early stop', flush=True)
                break
            b = blocks[bi].unsqueeze(0)
            lg = model(input_ids=b.to('cuda'), use_cache=False).logits
            s = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), b.to('cuda')[:, 1:].reshape(-1), reduction='sum').item()
            out.append({'sum': s, 'n': int(b[:, 1:].numel())})
    mean = sum(x['sum'] for x in out) / max(1, sum(x['n'] for x in out))
    return out, mean
_dp = hs_rows[0]
_d1 = _logprob_opts(_dp['ctx'], [' ' + e for e in _dp['endings'][:2]])
assert _d1 == _logprob_opts(_dp['ctx'], [' ' + e for e in _dp['endings'][:2]]), 'eval path not deterministic'
print('eval determinism ok', flush=True)
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
def _eval_hs(arm_key):
    base = V['rows']['hellaswag']
    assert len(hs_rows) == len(base)
    out = []
    correct = []
    for j, (r, b) in enumerate(zip(hs_rows, base)):
        if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
            print('HARD-GUARD: stopping HS early', flush=True)
            break
        opts = r['endings']
        ns = [tokenizer(o if o.startswith(' ') else ' ' + o, add_special_tokens=False)['input_ids'].__len__() for o in opts]
        assert r['label'] == b['label'] and len(opts) == b['n_opts'] and ns == b['opt_lens'], 'prompt drift: hellaswag idx %d' % j
        lps = _logprob_opts(r['ctx'], opts)
        pred = max(range(len(opts)), key=lambda i: lps[i])
        rec = dict(b)
        rec[arm_key] = {'logprobs': lps, 'pred': pred, 'correct': int(pred == r['label'])}
        out.append(rec)
        correct.append(int(pred == r['label']))
    return out, sum(correct) / max(1, len(correct)), correct
def _eval_lam(arm_key):
    base = V['rows']['lambada']
    assert len(lam_rows) == len(base)
    out = []
    correct = []
    nlls = []
    for j, (r, b) in enumerate(zip(lam_rows, base)):
        if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
            print('HARD-GUARD: stopping LAMBADA early', flush=True)
            break
        assert r['target'] == b['target'], 'prompt drift: lambada idx %d' % j
        nll, acc, nt = _lambada_score(r['context'], r['target'])
        rec = dict(b)
        rec[arm_key] = {'nll': nll, 'nll_per_tok': nll / max(1, nt), 'correct': acc, 'ntok': nt}
        out.append(rec)
        correct.append(acc)
        nlls.append(nll / max(1, nt))
    return out, sum(correct) / max(1, len(correct)), sum(nlls) / max(1, len(nlls)), correct, nlls
def _eval_hs_disabled(arm_key):
    base = V['rows']['hellaswag']
    assert len(hs_rows) == len(base)
    out = []
    correct = []
    with torch.inference_mode():
        inj.set_memory(None)
        for j, (r, b) in enumerate(zip(hs_rows, base)):
            if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
                print('HARD-GUARD: stopping disabled HS early', flush=True)
                break
            opts = r['endings']
            p = tokenizer(r['ctx'], return_tensors='pt', add_special_tokens=False)['input_ids']
            lps = []
            for c in opts:
                t = tokenizer(c if c.startswith(' ') else ' ' + c, return_tensors='pt', add_special_tokens=False)['input_ids']
                ids = torch.cat([p, t], dim=1)
                lg = model(input_ids=ids.to('cuda'), use_cache=False).logits.float()
                lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
                lps.append(lp.gather(1, t[0].to('cuda').unsqueeze(1)).sum().item())
            pred = max(range(len(opts)), key=lambda i: lps[i])
            rec = dict(b)
            rec[arm_key] = {'logprobs': lps, 'pred': pred, 'correct': int(pred == r['label'])}
            out.append(rec)
            correct.append(int(pred == r['label']))
    return out, sum(correct) / max(1, len(correct)), correct
def _eval_lam_disabled(arm_key):
    base = V['rows']['lambada']
    assert len(lam_rows) == len(base)
    out = []
    correct = []
    nlls = []
    with torch.inference_mode():
        inj.set_memory(None)
        for j, (r, b) in enumerate(zip(lam_rows, base)):
            if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
                print('HARD-GUARD: stopping disabled LAMBADA early', flush=True)
                break
            assert r['target'] == b['target'], 'prompt drift: lambada idx %d' % j
            p = tokenizer(r['context'], return_tensors='pt', add_special_tokens=False)['input_ids']
            t = tokenizer(r['target'] if r['target'].startswith(' ') else ' ' + r['target'], return_tensors='pt', add_special_tokens=False)['input_ids']
            ids = torch.cat([p, t], dim=1)
            lg = model(input_ids=ids.to('cuda'), use_cache=False).logits.float()
            lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
            nll = -lp.gather(1, t[0].to('cuda').unsqueeze(1)).sum().item()
            acc = int(bool((lp.argmax(-1).cpu() == t[0]).all()))
            rec = dict(b)
            rec[arm_key] = {'nll': nll, 'nll_per_tok': nll / max(1, int(t.shape[1])), 'correct': acc, 'ntok': int(t.shape[1])}
            out.append(rec)
            correct.append(acc)
            nlls.append(nll / max(1, int(t.shape[1])))
    return out, sum(correct) / max(1, len(correct)), sum(nlls) / max(1, len(nlls)), correct, nlls
for _task in ['hellaswag', 'lambada']:
    for a, b in [('real', 'disabled'), ('real-1m', 'real')]:
        va_a = [x[a]['correct'] for x in V['rows'][_task]]
        va_b = [x[b]['correct'] for x in V['rows'][_task]]
        m = _boot_diff(va_a, va_b)
        old = V['bootstrap'][_task]['%s-vs-%s' % (a, b)]
        assert abs(m['mean'] - old['mean']) < 1e-12 and abs(m['lo'] - old['lo']) < 1e-12 and abs(m['hi'] - old['hi']) < 1e-12, 'frozen mismatch: ' + _task
print('frozen v34 contrasts reproduced exactly (HS+LAMBADA, 1e-12)', flush=True)
_hs_dis, _hs_dis_acc, _hs_dis_c = _eval_hs_disabled('disabled-rescored')
_lam_dis, _lam_dis_acc, _lam_dis_nll, _lam_dis_c, _lam_dis_n = _eval_lam_disabled('disabled-rescored')
print('DISABLED rescored: HS %.4f | LAMBADA acc %.4f nll %.5f' % (_hs_dis_acc, _lam_dis_acc, _lam_dis_nll), flush=True)
_fro_hs = [x['disabled']['correct'] for x in V['rows']['hellaswag']]
_fro_lam = [x['disabled']['correct'] for x in V['rows']['lambada']]
assert abs(_hs_dis_acc - sum(_fro_hs) / len(_fro_hs)) < 1e-12, 'disabled HS drift vs frozen'
assert abs(_lam_dis_acc - sum(_fro_lam) / len(_fro_lam)) < 1e-12, 'disabled LAMBADA acc drift vs frozen'
(EOUT / 'hellaswag-disabled.jsonl').write_text('\n'.join(json.dumps(x) for x in _hs_dis))
(EOUT / 'lambada-disabled.jsonl').write_text('\n'.join(json.dumps(x) for x in _lam_dis))
_mark('disabled rescored (pass-through, matches frozen)')
_vd_dis_blocks, _vd_dis = _block_nll_disabled(val_full_chk)
_dom_dis = {}
for _d in DOMAINS:
    if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
        print('HARD-GUARD: stopping disabled domains early', flush=True)
        break
    _db, _dm = _block_nll_disabled(dom_blocks[_d])
    _dom_dis[_d] = _dm
print('DISABLED val %.5f | dom-mean %.5f' % (_vd_dis, sum(_dom_dis.values()) / max(1, len(_dom_dis))), flush=True)
_set_gamma_pair(1.0, 1.0)
_hs_raw, _hs_raw_acc, _hs_raw_c = _eval_hs('raw')
_lam_raw, _lam_raw_acc, _lam_raw_nll, _lam_raw_c, _lam_raw_n = _eval_lam('raw')
_v_raw_blocks, _v_raw = _block_nll(val_full_chk)
_dom_raw = {}
for _d in DOMAINS:
    if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
        print('HARD-GUARD: stopping raw domains early', flush=True)
        break
    _db, _dm = _block_nll(dom_blocks[_d])
    _dom_raw[_d] = _dm
print('RAW (1.0, 1.0): HS %.4f | LAMBADA acc %.4f nll %.5f | val %.5f' % (_hs_raw_acc, _lam_raw_acc, _lam_raw_nll, _v_raw), flush=True)
(EOUT / 'hellaswag-raw.jsonl').write_text('\n'.join(json.dumps(x) for x in _hs_raw))
(EOUT / 'lambada-raw.jsonl').write_text('\n'.join(json.dumps(x) for x in _lam_raw))
_restore_learned()
assert abs(float(inj.readers['2'].gamma.detach().cpu()) - orig_g2) < 1e-9 and abs(float(inj.readers['8'].gamma.detach().cpu()) - orig_g8) < 1e-9, 'gamma restore failed after raw'
_set_gamma_pair(STATIC_M2, STATIC_M8)
print('A static canonical in-memory gammas: IDX2=%.6f IDX8=%.6f' % (float(inj.readers['2'].gamma.detach().cpu()), float(inj.readers['8'].gamma.detach().cpu())), flush=True)
_hs_A, _hs_A_acc, _hs_A_c = _eval_hs('A-static')
_lam_A, _lam_A_acc, _lam_A_nll, _lam_A_c, _lam_A_n = _eval_lam('A-static')
_v_A_blocks, _v_A = _block_nll(val_full_chk)
_dom_A = {}
for _d in DOMAINS:
    if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
        print('HARD-GUARD: stopping A domains early', flush=True)
        break
    _db, _dm = _block_nll(dom_blocks[_d])
    _dom_A[_d] = _dm
print('A STATIC (1.25, 0.125): HS %.4f | LAMBADA acc %.4f nll %.5f | val %.5f | dom-mean %.5f' % (_hs_A_acc, _lam_A_acc, _lam_A_nll, _v_A, sum(_dom_A.values()) / max(1, len(_dom_A))), flush=True)
(EOUT / 'hellaswag-A.jsonl').write_text('\n'.join(json.dumps(x) for x in _hs_A))
(EOUT / 'lambada-A.jsonl').write_text('\n'.join(json.dumps(x) for x in _lam_A))
_restore_learned()
assert abs(float(inj.readers['2'].gamma.detach().cpu()) - orig_g2) < 1e-9 and abs(float(inj.readers['8'].gamma.detach().cpu()) - orig_g8) < 1e-9, 'gamma restore failed after A'
print('in-memory gammas restored to learned values after A; no reader files written', flush=True)
_mark('static arms done (disabled/raw/A)')
V40MNT = Path('/kaggle/input/qwen-ple-a2-final-v40')
V40_FILES = ['summary.json', 'hellaswag-calibrated.jsonl', 'lambada-calibrated.jsonl']
V40_SUMMARY_SHA = 'dd584526893d09956a600173286a64fc5c96dda5afe129463fc34bc5dfbebec1'
V40_TOL = 1e-6  # numerical tolerance for fresh-A vs persisted-v40 calibrated aggregates
def _fetch_v40():
    assert V40MNT.exists(), 'v40 dataset missing: attach ninnix/qwen-ple-a2-final-v40'
    for _f in V40_FILES + ['calibrated-reader.json']:
        assert (V40MNT / _f).exists(), 'missing v40 file: ' + _f
    h = hashlib.sha256()
    with open(V40MNT / 'summary.json', 'rb') as _fh:
        for _b in iter(lambda: _fh.read(1 << 20), b''):
            h.update(_b)
    assert h.hexdigest() == V40_SUMMARY_SHA, 'v40 summary bytes mismatch (wrong dataset version?)'
    print('v40: mounted dataset ok (4 files, summary SHA pinned)', flush=True)
    return V40MNT, {f: V40MNT / f for f in V40_FILES}
V40DIR, V40GOT = _fetch_v40()
# PRE-RUN INVARIANT 1: fresh static arm A must reproduce the persisted v40 calibrated results within numerical tolerance.
_v40sum = json.loads(V40GOT['summary.json'].read_text())
_v40cal = _v40sum['calibrated']
assert _v40cal['mult'] == [STATIC_M2, STATIC_M8] == [1.25, 0.125], 'v40 pin must be the (1.25, 0.125) canonical point'
_v40hs = [json.loads(x) for x in V40GOT['hellaswag-calibrated.jsonl'].read_text().splitlines() if x.strip()]
_v40lam = [json.loads(x) for x in V40GOT['lambada-calibrated.jsonl'].read_text().splitlines() if x.strip()]
assert len(_v40hs) == len(_hs_A) == 1000 and len(_v40lam) == len(_lam_A) == 1000, 'v40 row-count drift'
for _a, _b in zip(_v40hs, _hs_A):
    assert _a['idx'] == _b['idx'] and _a['label'] == _b['label'], 'v40 HS manifest drift'
for _a, _b in zip(_v40lam, _lam_A):
    assert _a['target'] == _b['target'], 'v40 LAMBADA manifest drift'
# v40 per-example correctness must match exactly (discrete decisions under deterministic inference)
assert [x['calibrated']['correct'] for x in _v40hs] == _hs_A_c, 'A HS decisions differ from v40 calibrated'
assert [x['calibrated']['correct'] for x in _v40lam] == _lam_A_c, 'A LAMBADA decisions differ from v40 calibrated'
for _k, _a, _b in [('hs_acc', _hs_A_acc, _v40cal['hs_acc']), ('lambada_acc', _lam_A_acc, _v40cal['lambada_acc']), ('lambada_nll', _lam_A_nll, _v40cal['lambada_nll_per_tok']), ('val_nll', _v_A, _v40cal['val_nll'])] + [('dom_' + _d, _dom_A[_d], _v40cal['domains'][_d]) for _d in DOMAINS]:
    assert abs(_a - _b) <= V40_TOL, 'A %s fresh=%.6f vs v40=%.6f beyond tolerance' % (_k, _a, _b)
    print('invariant1 %s: fresh-A %.6f vs v40 %.6f ok' % (_k, _a, _b), flush=True)
print('PRE-RUN INVARIANT 1 ok: static arm A reproduces persisted v40 calibrated results within tolerance', flush=True)
# PRE-RUN INVARIANT 2: dynamic arms B/C at init (before any optimizer step) must reproduce arm A.
# Gate part holds exactly (init alphas are constant 0.125/1.25, asserted to 1e-9/1e-6 below).
# The end-to-end path differs from the static-gamma path only by fp16-wrapper rounding
# (B scales the fp16-cast reader delta; A scales in fp32 pre-cast), which RMSNorm/attention
# and max-over-vocab amplify to ~0.07 worst-case (measured 0.0664) while means wash out.
# So: exact gate asserts + loose wiring-disaster bounds end-to-end (a dead/miswired hook
# shifts logits O(1+) and NLL-means O(1e-2+); fp noise stays far below both).
INIT_TOL = 2.5e-1  # max|logit diff| wiring bound; fp16-wrapper noise floor measured 0.066
INIT_NLL_TOL = 5e-3  # probe NLL-mean bound; fp-noise mean is ~1e-4, systematic mis-scale >=1e-2
_probe_ids = torch.cat([calib_blocks[:4], val_full_chk[:2], dom_blocks['general'][:1]], dim=0)
assert _probe_ids.shape == (7, 512), 'probe shape drift'
_set_gamma_pair(STATIC_M2, STATIC_M8)
with torch.inference_mode():
    inj.set_memory(_cg.lookup(addresses(_probe_ids.cpu())).to('cuda'))
    _lgA = model(input_ids=_probe_ids.to('cuda'), use_cache=False).logits.float().cpu()
_restore_learned()
inj.set_memory(None)
for _tag, _fix in [('B', 1.25), ('C', None)]:
    _arb0 = MemoryArbitration(hidden=C.HIDDEN).to('cuda')
    assert abs(float(_arb0.alpha2().detach()) - 1.25) < 1e-9, 'init alpha2 must be 1.25'
    _arb0.requires_grad_(False)
    _hk0 = ArbHooks(model, inj, _arb0)
    if _fix is not None:
        _hk0.fixed_alpha2 = _fix
    with torch.inference_mode():
        _hk0.set_memory(_cg.lookup(addresses(_probe_ids.cpu())).to('cuda'))
        _lg0 = model(input_ids=_probe_ids.to('cuda'), use_cache=False).logits.float().cpu()
        _a80 = _hk0.last_alpha8.float().reshape(-1)
    _hk0.close()
    assert float(_a80.std(correction=0)) < 1e-9 and abs(float(_a80.mean()) - 0.125) < 1e-6, 'init alpha8_t must be constant 0.125'
    _dmax = (_lgA - _lg0).abs().max().item()
    _nllA = F.cross_entropy(_lgA[:, :-1].reshape(-1, _lgA.shape[-1]), _probe_ids[:, 1:].reshape(-1)).item()
    _nll0 = F.cross_entropy(_lg0[:, :-1].reshape(-1, _lg0.shape[-1]), _probe_ids[:, 1:].reshape(-1)).item()
    assert _dmax < INIT_TOL, 'init-%s max|diff|=%g exceeds tolerance' % (_tag, _dmax)
    assert abs(_nllA - _nll0) < INIT_NLL_TOL, 'init-%s NLL-mean diff exceeds tolerance' % _tag
    print('PRE-RUN INVARIANT 2 ok: init-%s reproduces A (gates exact, max|diff|=%.2e, dNLL=%.2e over %d tokens)' % (_tag, _dmax, abs(_nllA - _nll0), _probe_ids.numel()), flush=True)
inj.set_memory(None)
_mark('pre-run invariants ok (A==v40, init-B/C==A)')
resB = train_arb('b', learn_alpha2=False)
resC = train_arb('c', learn_alpha2=True)
_mark('arbitration trained (B fixed-IDX2, C learned-IDX2)')
def _run_arb_arm(tag, state, learn_alpha2, arm_key):
    arb = MemoryArbitration(hidden=C.HIDDEN).to('cuda')
    arb.w.data.copy_(state['w'].to('cuda'))
    arb.b.data.copy_(state['b'].to('cuda'))
    arb.alpha2_raw.data.copy_(state['alpha2_raw'].to('cuda'))
    arb.requires_grad_(False)
    model.requires_grad_(False)
    inj.requires_grad_(False)
    hooks = ArbHooks(model, inj, arb)
    if not learn_alpha2:
        hooks.fixed_alpha2 = 1.25
    hooks.set_memory(None)
    a8_collect = {'hellaswag': [], 'lambada': [], 'full-val': [], 'general': [], 'code': [], 'math': [], 'scientific': [], 'multilingual': []}
    def _lp_arb(prompt, conts, bucket):
        p = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
        outs = []
        with torch.inference_mode():
            for c in conts:
                t = tokenizer(c if c.startswith(' ') else ' ' + c, return_tensors='pt', add_special_tokens=False)['input_ids']
                ids = torch.cat([p, t], dim=1)
                hooks.set_memory(_cg.lookup(addresses(ids)).to('cuda'))
                lg = model(input_ids=ids.to('cuda'), use_cache=False).logits.float()
                a8_collect[bucket].append(hooks.last_alpha8.reshape(-1).cpu())
                lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
                outs.append(lp.gather(1, t[0].to('cuda').unsqueeze(1)).sum().item())
        return outs
    def _lam_arb(context, target):
        p = tokenizer(context, return_tensors='pt', add_special_tokens=False)['input_ids']
        t = tokenizer(target if target.startswith(' ') else ' ' + target, return_tensors='pt', add_special_tokens=False)['input_ids']
        ids = torch.cat([p, t], dim=1)
        with torch.inference_mode():
            hooks.set_memory(_cg.lookup(addresses(ids)).to('cuda'))
            lg = model(input_ids=ids.to('cuda'), use_cache=False).logits.float()
            a8_collect['lambada'].append(hooks.last_alpha8.reshape(-1).cpu())
            lp = torch.log_softmax(lg[0, p.shape[1] - 1:-1], dim=-1)
            nll = -lp.gather(1, t[0].to('cuda').unsqueeze(1)).sum().item()
            acc = int(bool((lp.argmax(-1).cpu() == t[0]).all()))
        return nll, acc, int(t.shape[1])
    def _blk_arb(blocks, bucket):
        out = []
        with torch.inference_mode():
            for bi in range(blocks.shape[0]):
                if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
                    print('HARD-GUARD: arb block NLL early stop', flush=True)
                    break
                b = blocks[bi].unsqueeze(0)
                hooks.set_memory(_cg.lookup(addresses(b.cpu())).to('cuda'))
                lg = model(input_ids=b.to('cuda'), use_cache=False).logits
                a8_collect[bucket].append(hooks.last_alpha8.reshape(-1).cpu())
                s = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), b.to('cuda')[:, 1:].reshape(-1), reduction='sum').item()
                out.append({'sum': s, 'n': int(b[:, 1:].numel())})
        mean = sum(x['sum'] for x in out) / max(1, sum(x['n'] for x in out))
        return out, mean
    base_hs = V['rows']['hellaswag']
    hs_out = []; hs_corr = []
    for j, (r, b) in enumerate(zip(hs_rows, base_hs)):
        if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
            print('HARD-GUARD: stopping arb HS early', flush=True)
            break
        assert r['label'] == b['label'], 'prompt drift HS'
        lps = _lp_arb(r['ctx'], r['endings'], 'hellaswag')
        pred = max(range(len(r['endings'])), key=lambda i: lps[i])
        rec = dict(b); rec[arm_key] = {'logprobs': lps, 'pred': pred, 'correct': int(pred == r['label'])}
        hs_out.append(rec); hs_corr.append(int(pred == r['label']))
    hs_acc = sum(hs_corr) / max(1, len(hs_corr))
    base_lam = V['rows']['lambada']
    lam_out = []; lam_corr = []; lam_n = []
    for j, (r, b) in enumerate(zip(lam_rows, base_lam)):
        if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
            print('HARD-GUARD: stopping arb LAMBADA early', flush=True)
            break
        assert r['target'] == b['target'], 'prompt drift LAMBADA'
        nll, acc, nt = _lam_arb(r['context'], r['target'])
        rec = dict(b); rec[arm_key] = {'nll': nll, 'nll_per_tok': nll / max(1, nt), 'correct': acc, 'ntok': nt}
        lam_out.append(rec); lam_corr.append(acc); lam_n.append(nll / max(1, nt))
    lam_acc = sum(lam_corr) / max(1, len(lam_corr)); lam_nll = sum(lam_n) / max(1, len(lam_n))
    _vb, _v = _blk_arb(val_full_chk, 'full-val')
    _dm = {}
    for _d in DOMAINS:
        if (time.perf_counter() - T0) > C.EVAL_HARD_S - 1800:
            print('HARD-GUARD: stopping arb domains early', flush=True)
            break
        _db, _dd = _blk_arb(dom_blocks[_d], _d)
        _dm[_d] = _dd
    stats = {}
    for k, vs in a8_collect.items():
        if not vs:
            continue
        stats[k] = _alpha_stats(torch.cat(vs))
    hooks.close()
    (EOUT / f'hellaswag-{tag}.jsonl').write_text('\n'.join(json.dumps(x) for x in hs_out))
    (EOUT / f'lambada-{tag}.jsonl').write_text('\n'.join(json.dumps(x) for x in lam_out))
    print('%s: HS %.4f | LAMBADA acc %.4f nll %.5f | val %.5f | dom-mean %.5f | mean_a8 full-val %.4f' % (tag, hs_acc, lam_acc, lam_nll, _v, sum(_dm.values()) / max(1, len(_dm)), stats.get('full-val', {}).get('mean', 0)), flush=True)
    return {'hs': (hs_out, hs_acc, hs_corr), 'lam': (lam_out, lam_acc, lam_nll, lam_corr, lam_n), 'val': _v, 'val_blocks': _vb, 'doms': _dm, 'alpha_stats': stats}
evalB = _run_arb_arm('B', resB['arb_state'], False, 'B-dynamic')
evalC = _run_arb_arm('C', resC['arb_state'], True, 'C-dynamic-alpha2')
_mark('dynamic arms done (B fixed-IDX2, C learned-IDX2)')
alpha_report = {
    'B': {'alpha2': 1.25, 'learn_alpha2': False, 'alpha8_by_dataset': evalB['alpha_stats']},
    'C': {'alpha2': resC['alpha2'], 'learn_alpha2': True, 'alpha8_by_dataset': evalC['alpha_stats']},
}
(EOUT / 'alpha8-stats.json').write_text(json.dumps(alpha_report, indent=2))
for _tag2, _ev in [('B', evalB), ('C', evalC)]:
    for _ds, _st in _ev['alpha_stats'].items():
        print('alpha8 %s/%s: mean %.4f std %.4f p05 %.4f p50 %.4f p95 %.4f n=%d' % (_tag2, _ds, _st['mean'], _st['std'], _st['p05'], _st['p50'], _st['p95'], _st['n']), flush=True)
print('learned alpha2: B fixed 1.2500 | C %.4f in [0.75,1.75]' % resC['alpha2'], flush=True)
boot = {}
for _label, _pair in [('B-vs-A', (evalB['hs'][2], _hs_A_c)), ('C-vs-A', (evalC['hs'][2], _hs_A_c)), ('C-vs-B', (evalC['hs'][2], evalB['hs'][2])), ('B-vs-disabled', (evalB['hs'][2], _hs_dis_c)), ('C-vs-disabled', (evalC['hs'][2], _hs_dis_c))]:
    m = _boot_diff(_pair[0], _pair[1])
    boot['hs-' + _label] = m
    print('HS %s: dacc %+.4f p=%.4g' % (_label, m['mean'], m['p']), flush=True)
for _label, _pair in [('B-vs-A', (evalB['lam'][3], _lam_A_c)), ('C-vs-A', (evalC['lam'][3], _lam_A_c)), ('C-vs-B', (evalC['lam'][3], evalB['lam'][3]))]:
    m = _boot_diff(_pair[0], _pair[1])
    boot['lambada-acc-' + _label] = m
    print('LAMBADA-acc %s: dacc %+.4f p=%.4g' % (_label, m['mean'], m['p']), flush=True)
for _label, _pair in [('B-vs-A', (evalB['lam'][4], _lam_A_n)), ('C-vs-A', (evalC['lam'][4], _lam_A_n)), ('C-vs-B', (evalC['lam'][4], evalB['lam'][4]))]:
    m = _boot_diff(_pair[0], _pair[1])
    boot['lambada-nll-' + _label] = m
    print('LAMBADA-nll %s: dnll %+.5f p=%.4g' % (_label, m['mean'], m['p']), flush=True)
(EOUT / 'bootstrap.json').write_text(json.dumps(boot, indent=2))
print('contrasts added (B/C vs A static canonical, paired)', flush=True)
def _retain(cal, raw, dis):
    _den = (raw - dis)
    if abs(_den) < 1e-12:
        return None
    return (cal - dis) / _den
pareto = {
    'val_nll': {'A': _retain(_v_A, _v_raw, _vd_dis), 'B': _retain(evalB['val'], _v_raw, _vd_dis), 'C': _retain(evalC['val'], _v_raw, _vd_dis)},
    'mean_domain_nll': {'A': _retain(sum(_dom_A.values()) / len(_dom_A), sum(_dom_raw.values()) / len(_dom_raw), sum(_dom_dis.values()) / len(_dom_dis)), 'B': _retain(sum(evalB['doms'].values()) / len(evalB['doms']), sum(_dom_raw.values()) / len(_dom_raw), sum(_dom_dis.values()) / len(_dom_dis)), 'C': _retain(sum(evalC['doms'].values()) / len(evalC['doms']), sum(_dom_raw.values()) / len(_dom_raw), sum(_dom_dis.values()) / len(_dom_dis))},
    'hs_acc': {'A': _hs_A_acc, 'B': evalB['hs'][1], 'C': evalC['hs'][1]},
    'lambada_acc': {'A': _lam_A_acc, 'B': evalB['lam'][1], 'C': evalC['lam'][1]},
    'lambada_nll': {'A': _lam_A_nll, 'B': evalB['lam'][2], 'C': evalC['lam'][2]},
}
for _k, _v in pareto.items():
    print('pareto %s: A=%s B=%s C=%s' % (_k, str(_v['A'])[:7] if _v['A'] is not None else 'n/a', str(_v['B'])[:7] if _v['B'] is not None else 'n/a', str(_v['C'])[:7] if _v['C'] is not None else 'n/a'), flush=True)
(EOUT / 'config.json').write_text(json.dumps({'seeds': {'eval': C.EVAL_SEED, 'ple': C.SEED, 'arb': ARB_SEED}, 'batch': C.EVAL_BS, 'decoding': 'none; teacher-forced logprob only', 'precision': 'backbone-fp16 reader-FP32 ple-FP32 arb-FP32', 'arms_new': ['disabled-rescored', 'real-5m-raw', 'A-static-1p25-0p125', 'B-dynamic-fixed', 'C-dynamic-alpha2'], 'arms_frozen': ['disabled', 'random', 'permuted', 'real', 'real-1m'], 'frozen_ref': REF, 'arbitration': {'alpha2_init': ALPHA2_INIT, 'alpha2_range': [ALPHA2_LO, ALPHA2_LO + ALPHA2_SPAN], 'alpha8_init': ALPHA8_INIT, 'b_init': B_INIT, 'params': 1026, 'lr': ARB_LR, 'wd': ARB_WD, 'warm': ARB_WARM, 'lambda_mean_reg': ARB_LAMBDA, 'tokens': ARB_TOTAL, 'corpus': list(ARB_SOURCES)}, 'val': 'frozen full-val (1024 blocks)', 'domains': 'five frozen Stage-A2 artifacts (64 blocks each)', 'note': 'tiny arbitration only; backbone/PLE/W_K/W_V/beta/gamma frozen; gammas scaled in-memory and restored; B/C differ only in learned alpha2; no R4'}, indent=2))
(EOUT / 'checkpoint-shas.json').write_text(json.dumps(_cksha, indent=2))
(EOUT / 'timings.json').write_text(json.dumps(timings, indent=2))
summ = {
    'disabled': {'hs_acc': _hs_dis_acc, 'lambada_acc': _lam_dis_acc, 'lambada_nll_per_tok': _lam_dis_nll, 'val_nll': _vd_dis, 'domains': _dom_dis},
    'raw': {'hs_acc': _hs_raw_acc, 'lambada_acc': _lam_raw_acc, 'lambada_nll_per_tok': _lam_raw_nll, 'val_nll': _v_raw, 'domains': _dom_raw, 'mult': [1.0, 1.0]},
    'A-static': {'hs_acc': _hs_A_acc, 'lambada_acc': _lam_A_acc, 'lambada_nll_per_tok': _lam_A_nll, 'val_nll': _v_A, 'domains': _dom_A, 'mult': [STATIC_M2, STATIC_M8]},
    'B-dynamic-fixed': {'hs_acc': evalB['hs'][1], 'lambada_acc': evalB['lam'][1], 'lambada_nll_per_tok': evalB['lam'][2], 'val_nll': evalB['val'], 'domains': evalB['doms'], 'alpha2': 1.25},
    'C-dynamic-alpha2': {'hs_acc': evalC['hs'][1], 'lambada_acc': evalC['lam'][1], 'lambada_nll_per_tok': evalC['lam'][2], 'val_nll': evalC['val'], 'domains': evalC['doms'], 'alpha2': resC['alpha2']},
    'pareto_vs_disabled': pareto,
    'success': 'B/C improve static canonical Pareto point iff they retain more raw val/domain gain without reintroducing LAMBADA regression',
}
(EOUT / 'summary.json').write_text(json.dumps(summ, indent=2))
inj.close()
_mark('STAGE A2-ARB COMPLETE: A/B/C + alpha8 distributions + learned alpha2; backbone/PLE/reader weights unchanged; no R4')
print('Tiny arbitration only (w, b, alpha2_raw); no backbone/PLE/W_K/W_V/beta/gamma updates. No R4.', flush=True)
