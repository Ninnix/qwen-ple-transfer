# GENERATED from kaggle_qwen35_08b_ple_eval_a2_arb.ipynb sha=e8eb4a69634ccc75a78ca84aa02f74beb38d40e0d89a56f4afc04e68aac041fc cells=c-hashing,c-reader,c-inject,c-ple. Do not hand-edit; rerun sync_frozen.py.
# --- %s (verbatim) ---
# Cell 3 — EXACT source addressing (port of src/qwen36_ple/hashing.py — do not modify)
import math
import torch

MASK64=(1<<64)-1; GAMMA=0x9E3779B97F4A7C15; M1=0xBF58476D1CE4E5B9; M2=0x94D049BB133111EB; LPRIME=10007

def splitmix64(v):
    v=(v+GAMMA)&MASK64; v=((v^(v>>30))*M1)&MASK64; v=((v^(v>>27))*M2)&MASK64; return (v^(v>>31))&MASK64

def layer_multipliers(vocab, ngram, ple_idx=0, seed=1234):
    mmax=((1<<63)-1)//max(vocab,1); hb=max(1,mmax//2); base=seed+LPRIME*ple_idx
    return tuple(2*(splitmix64((base+GAMMA*(i+1))&MASK64)%hb)+1 for i in range(ngram))

def is_prime(v):
    if v<2: return False
    if v%2==0: return v==2
    return all(v%d for d in range(3, math.isqrt(v)+1, 2))

def nth_prime_after(s, k):
    p=s
    for _ in range(k):
        p+=1
        while not is_prime(p): p+=1
    return p

def head_layout(ngram=3, hpn=8, base=20_000_000, ple_idx=0):
    n=(ngram-1)*hpn
    sizes=tuple(nth_prime_after(base-1, ple_idx*n+h+1) for h in range(n))
    off=[]; t=0
    for s in sizes: off.append(t); t+=s
    return sizes, tuple(off)

def shift_right_ignore_eos(ids, shift, eos):
    if shift==0: return ids
    B,L=ids.shape; pos=torch.arange(L, device=ids.device)
    eos_pos=torch.where(ids==eos, pos, -1); prev_inc=torch.cummax(eos_pos,1).values
    prev=torch.cat([eos_pos.new_full((B,1),-1), prev_inc[:,:-1]],1)
    inseg=pos.unsqueeze(0)-(prev+1); src=pos-shift
    sh=ids.gather(1, src.clamp_min(0).unsqueeze(0).expand(B,-1))
    valid=(inseg>=shift)&(src.unsqueeze(0)>=0)
    return torch.where(valid, sh, ids.new_full((), eos))

def ngram_indices(ids, eos_token_id=248044, vocab_size=248320, ngram_size=3, heads_per_ngram=8,
                    vocab_size_base=20_000_000, ple_layer_index=0, seed=1234):
    ids=ids.long()
    mult=torch.tensor(layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed), device=ids.device)
    sizes, offs=head_layout(ngram_size, heads_per_ngram, vocab_size_base, ple_layer_index)
    sizes=torch.tensor(sizes, device=ids.device); offs=torch.tensor(offs, device=ids.device)
    sh=[shift_right_ignore_eos(ids,s,eos_token_id) for s in range(ngram_size)]
    blocks=[]
    for ng in range(2, ngram_size+1):
        st=(ng-2)*heads_per_ngram; mixed=sh[0]*mult[0]
        for p in range(1,ng): mixed=torch.bitwise_xor(mixed, sh[p]*mult[p])
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[st:st+heads_per_ngram])+offs[st:st+heads_per_ngram])
    return torch.cat(blocks,-1)  # [B,L,16] GLOBAL PLE addresses, never raw token ids

_a=ngram_indices(torch.tensor([[1,2,3,4,5]])); _b=ngram_indices(torch.tensor([[1,2,3,4,5]]))
assert torch.equal(_a,_b) and _a.shape==(1,5,16)
SIZES, OFFS = head_layout()
print('hash ok; slots/head addrs e.g.', tuple(_a[0,2,:4].tolist()), '| head0 range', (OFFS[0], OFFS[0]+SIZES[0]))
def addresses(token_cpu):
    '''ONLY path from tokens to PLE rows: exact source hashes -> global head addresses.'''
    return ngram_indices(token_cpu, eos_token_id=C.EOS, vocab_size=C.VOCAB,
                         ngram_size=C.NGRAM, heads_per_ngram=C.HEADS_PER_NGRAM,
                         vocab_size_base=C.VOCAB_BASE, ple_layer_index=0, seed=C.SEED)
# --- %s (verbatim) ---
# Cell 4 — shared-value reader (hidden=1024; reductions stay FP32 so fp16 backbone is safe)
import math
import torch
from torch import nn

def rms_norm(x, eps=1e-6):  # always FP32 reduction, cast back: fp16-safe
    return x.float().mul(torch.rsqrt(x.float().square().mean(-1, keepdim=True)+eps)).to(x.dtype)

class SharedValueReader(nn.Module):
    def __init__(self, mem_dim=2560, hidden=1024, branches=1, gamma=0.0):
        super().__init__(); self.mem_dim=mem_dim; self.hidden=hidden; self.branches=branches
        self.keys=nn.ModuleList(nn.Linear(mem_dim, hidden, bias=False) for _ in range(branches))
        self.value=nn.Linear(mem_dim, hidden, bias=False)
        self.beta=nn.Parameter(torch.zeros(branches))
        self.gamma=nn.Parameter(torch.tensor(float(gamma)))
        self.last_gate=None
    def stats(self):
        if self.last_gate is None: return None
        g=self.last_gate.float()
        return {'mean':g.mean().item(),'std':g.std(correction=0).item(),'near_zero':(g<0.01).float().mean().item()}
    def forward(self, h, m):
        assert h.shape[:-1]==m.shape[:-1], (h.shape, m.shape)
        h_dtype=h.dtype; h=h.float(); m=m.float()  # backbone may be fp16; reader computes FP32
        q=rms_norm(h); v=self.value(m); gs=[]
        for b,proj in enumerate(self.keys):
            k=rms_norm(proj(m))
            s=(q.float()*k.float()).sum(-1)/math.sqrt(self.hidden)
            gs.append(torch.sigmoid(s+self.beta[b].float()).to(v.dtype))
        g=torch.stack(gs,0)
        o=(g.unsqueeze(-1)*v.unsqueeze(0)).mean(0)
        self.last_gate=g.detach()
        return (h+self.gamma*o).to(h_dtype)  # residual back to backbone dtype; params stay FP32

_r=SharedValueReader(gamma=0.0); _h=torch.randn(1,4,1024); _m=torch.randn(1,4,2560)
assert torch.equal(_r(_h,_m),_h)
print('reader identity ok; R=1 params:', sum(p.numel() for p in _r.parameters()))
# --- %s (verbatim) ---
# Cell 5 — injection hooks (IDX convention) + layer helper
import torch
from torch import nn

def decoder_layers(model):
    for path in ['model.layers','language_model.layers','transformer.h']:
        o=model
        try:
            for a in path.split('.'): o=getattr(o,a)
            if len(o)==24 or len(o)>0: return o
        except Exception: pass
    raise RuntimeError('decoder layers not found')

class ReaderInjection(nn.Module):
    '''layers = zero-based IDX list, exactly like the 35B run (e.g. (2,) = third block).'''
    def __init__(self, model, layers, mem_dim=2560, hidden=1024, branches=1, gamma=0.0):
        super().__init__()
        self.idx=tuple(layers)
        self.readers=nn.ModuleDict({str(l):SharedValueReader(mem_dim,hidden,branches,gamma) for l in layers})
        self.memory=None; self.handles=[]
        dec=decoder_layers(model)
        assert len(dec)==C.N_LAYERS, f'decoder count {len(dec)} != {C.N_LAYERS} — wrong hook target'
        for l in layers:
            self.handles.append(dec[l].register_forward_pre_hook(self._hook(str(l)), with_kwargs=True))
        print(f'inject at IDX {list(layers)} = HUMAN {[l+1 for l in layers]}')
    def _hook(self,name):
        def fn(mod,args,kw):
            if self.memory is None: return args,kw
            m=self.memory
            L=args[0].shape[1] if args else kw['hidden_states'].shape[1]
            if m.shape[1]!=L: m=m[:,:L]
            if args: return (self.readers[name](args[0],m),*args[1:]),kw
            kw['hidden_states']=self.readers[name](kw['hidden_states'],m); return args,kw
        return fn
    def set_memory(self,m): self.memory=m
    def close(self):
        [h.remove() for h in self.handles]; self.handles.clear()

print('injection ok')
# --- %s (verbatim) ---
# Cell 6 — PLE stores: real (mount-only, exact scale or abort) + calibrated controls
import json, os
from pathlib import Path
import torch
from safetensors import safe_open

PLE_TMPL='model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{p}.weight'
PLE_SCALE='model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale'

def rss_mb():
    '''Host RSS in MiB (Linux /proc; -1 if unavailable). Proves bounded RAM.'''
    try:
        with open('/proc/self/status') as _f:
            for _line in _f:
                if _line.startswith('VmRSS:'): return float(_line.split()[1])/1024
    except Exception: pass
    try:
        import resource; return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except Exception: return -1.0

def find_ple_manifests():
    hits=sorted(Path('/kaggle/input').glob('*/manifest.json'))+sorted(Path('/kaggle/input').glob('*/*/manifest.json'))
    out=[]
    for h in hits:
        try:
            m=json.loads(h.read_text())
            if isinstance(m, dict) and 'parts' in m: out.append(h)
        except Exception: pass
    return out

class MountPLE:
    '''Real frozen PLE (row-level mmap, 35B-validated). REQUIRES /kaggle/input mount. Never downloads. Never caches parts: each lookup fetches ONLY needed rows via safetensors get_slice runs, so host RAM stays bounded after all 128 parts are touched.'''
    def __init__(self, manifest=None):
        manifests=[Path(manifest)] if manifest else find_ple_manifests()
        if not manifests or not all(p.exists() for p in manifests):
            raise RuntimeError('Real-PLE ABORT: no /kaggle/input PLE dataset attached. Attach the pinned shards+manifest.json dataset(s) first; refusing to download 48.7 GiB into /kaggle/working.')
        self.part_paths={}; revs=set()
        for mp in manifests:
            m=json.loads(mp.read_text())
            if not revs: self.rpp=m.get('rows_per_part',2500012); self.rd=m.get('row_dim',160)
            revs.add(m.get('ple_revision','unknown'))
            for k,v in m['parts'].items():
                p=mp.parent/v
                if not p.exists(): continue  # each dataset holds only its own shards
                if int(k) in self.part_paths:
                    assert self.part_paths[int(k)].name==p.name, f'part {k} filename clash'
                    continue
                self.part_paths[int(k)]=p
        assert len(revs)==1, f'mixed PLE revisions: {revs}'
        self.ple_revision=revs.pop()
        assert len(self.part_paths)==128, f'need all 128 parts, have {len(self.part_paths)}'
        missing=[str(p) for p in self.part_paths.values() if not p.exists()]
        if missing: raise RuntimeError(f"Real-PLE ABORT: {len(missing)} shard files missing, e.g. {missing[0]}")
        self.scale=self._resolve_scale()  # exact scale or abort — no fallback constant
        self.calls=0; self.rows_read=0
        self.parts_touched=set()  # cumulative DISTINCT parts; no part tensors ever held
        print(f'mounted PLE rev={self.ple_revision} parts={len(self.part_paths)} scale={self.scale}')
    def _resolve_scale(self):
        for f in sorted(set(self.part_paths.values())):
            try:
                with safe_open(f, framework='pt', device='cpu') as fh:
                    if PLE_SCALE in fh.keys():
                        return float(fh.get_tensor(PLE_SCALE).float().mean())
            except Exception: pass
        raise RuntimeError('Real-PLE ABORT: weight_scale tensor not found in pinned shards/index. Refusing hardcoded fallback.')
    def stats(self):
        return {'calls': self.calls, 'rows_read': self.rows_read,
                'parts_touched': len(self.parts_touched), 'held_part_tensors': 0,
                'scale': self.scale, 'rss_MiB': round(rss_mb(), 1)}
    @staticmethod
    def tensor_name(part): return PLE_TMPL.format(p=part)
    def lookup(self, indices):  # indices = GLOBAL head addresses [..,16] from ngram_indices()
        '''Row-level mmap reads: group deduped addresses by part (sorted), fetch ONLY
        needed rows as contiguous get_slice runs, dequantize the gathered rows. Full
        part tensors (~381 MiB each) are never materialized or cached.'''
        shape=indices.shape; flat=indices.detach().cpu().long().reshape(-1)
        uniq, inv=torch.unique(flat, return_inverse=True)  # dedup repeated addresses
        parts=torch.div(uniq, self.rpp, rounding_mode='floor'); local=uniq%self.rpp
        table=torch.empty(uniq.numel(), self.rd, dtype=torch.float32)
        for p in torch.unique(parts).tolist():  # ascending part order (page-friendly)
            pos=torch.nonzero(parts==p).flatten()
            lrows=local.index_select(0, pos); srows, sidx=torch.sort(lrows)
            runs=[]; a=int(srows[0]); prev=a
            for r in srows[1:].tolist():
                if r==prev+1: prev=r
                else: runs.append((a, prev+1)); a=prev=r
            runs.append((a, prev+1))
            path=self.part_paths.get(p)
            if path is None: raise FileNotFoundError(f'PLE part {p} not in manifest')
            with safe_open(str(path), framework='pt', device='cpu') as fh:
                sl=fh.get_slice(self.tensor_name(p))
                got=torch.cat([sl[x:y].to(torch.float32) for x, y in runs])*self.scale
            table.index_copy_(0, pos.index_select(0, sidx), got)  # got aligns with srows
        self.calls+=1; self.rows_read+=uniq.numel(); self.parts_touched.update(torch.unique(parts).tolist())
        return table[inv].reshape(*shape, self.rd).flatten(-2)  # [..,16,160]->[..,2560]

def permute_addresses(addrs, seed=777):
    '''Deterministic per-head bijective address permutation (shared by builder + store).'''
    sizes, offs=head_layout()
    S=torch.tensor(sizes); O=torch.tensor(offs)
    A=[]; B=[]
    for h,(s,o) in enumerate(zip(sizes, offs)):
        a=int(splitmix64((seed+10007*(h+1))&MASK64)%(s-1))+1
        b=int(splitmix64(((seed^0x9E3779B97F4A7C15)+7919*(h+1))&MASK64)%s)
        assert a%s!=0, 'A must be coprime to prime head size'
        A.append(a); B.append(b)
    A=torch.tensor(A); B=torch.tensor(B)
    f=addrs.long().cpu()
    H=f.shape[-1]
    O=O[:H]; A=A[:H]; B=B[:H]; S=S[:H]
    return O+((f-O)*A+B)%S

class RandomPLE:
    '''Calibrated per-head deterministic control: same global address -> same 160-d row, every call.
    Means/stds are per-head scalars measured from real rows in the frozen working set.
    16 rows concatenate to 2560-d. No 0.06 fallback.'''
    def __init__(self, head_means, head_stds, seed=0, row_dim=160):
        assert len(head_means)==16 and len(head_stds)==16, 'need 16 per-head stats'
        self.means=[float(m) for m in head_means]
        self.stds=[float(s) for s in head_stds]
        assert all(s>0 for s in self.stds), 'stds must be positive (calibrated)'
        self.seed=seed; self.rd=row_dim
        _sizes,_offs=head_layout()
        self._S=list(_sizes); self._O=list(_offs)
    def _head_of(self, a):
        for h in range(16):
            if self._O[h]<=a<self._O[h]+self._S[h]: return h
        raise ValueError('address outside head ranges')
    def _rows_for(self, uniq, heads):
        rows=[]
        for a,h in zip(uniq.tolist(), heads.tolist()):
            g=torch.Generator(); g.manual_seed((self.seed*1000003+int(a))%2**63)
            rows.append(torch.randn(self.rd, generator=g)*self.stds[h]+self.means[h])
        return torch.stack(rows)
    def lookup(self, indices):
        shape=indices.shape; flat=indices.detach().cpu().long().reshape(-1)
        uniq, inv=torch.unique(flat, return_inverse=True)
        heads=torch.tensor([self._head_of(int(a)) for a in uniq.tolist()])
        table=self._rows_for(uniq, heads)
        return table[inv].reshape(*shape, self.rd).flatten(-2)

def calibrate_head_stats(compact, n_per_head=4096, seed=0):
    '''Measure per-head mean/std from representative real rows in the compact working set.
    Samples n_per_head rows per head from the frozen prefix (train + full-val).'''
    sizes, offs=head_layout()
    addrs=compact.addrs.to(torch.int64)
    g=torch.Generator().manual_seed(seed)
    means=[]; stds=[]
    for h in range(16):
        lo=offs[h]; hi=offs[h]+sizes[h]
        pos=torch.nonzero((addrs>=lo)&(addrs<hi)).flatten()
        assert len(pos)>=n_per_head, f'head {h} only {len(pos)} rows'
        pick=pos[torch.randint(0,len(pos),(n_per_head,),generator=g)]
        rows=compact.rows.index_select(0, pick).to(torch.float32)*compact.scale
        means.append(float(rows.mean()))
        stds.append(float(rows.std(correction=0)))
    print('calibrated head means:', [round(m,6) for m in means])
    print('calibrated head stds:', [round(s,6) for s in stds])
    return means, stds

class PermutedPLE:
    '''Bijective per-head permutation: off_h + ((a-off_h)*A_h + B_h) % size_h.
    Preserves head ranges (sizes are prime, A_h % size_h != 0 so gcd=1 i.e. coprime) and table distribution.
    Wraps a compact working-set cache, never /kaggle/input random reads during training.'''
    def __init__(self, base, seed=777):
        self.b=base; self.seed=seed
        sizes, offs=head_layout()
        self.S=torch.tensor(sizes); self.O=torch.tensor(offs)
        A=[]; B=[]
        for h,(s,o) in enumerate(zip(sizes, offs)):
            a=int(splitmix64((seed+10007*(h+1))&MASK64)%(s-1))+1
            b=int(splitmix64(((seed^0x9E3779B97F4A7C15)+7919*(h+1))&MASK64)%s)
            assert a%s!=0, 'A must be coprime to prime head size'
            A.append(a); B.append(b)
        self.A=torch.tensor(A); self.B=torch.tensor(B)
    def lookup(self, indices):
        H=indices.shape[-1]; f=indices.long().cpu()
        O=self.O[:H]; A=self.A[:H]; B=self.B[:H]; S=self.S[:H]
        return self.b.lookup((O+((f-O)*A+B)%S).to(indices.device) if indices.is_cuda else (O+((f-O)*A+B)%S))

class CompactPLE:
    '''Compact working set for one frozen token prefix: sorted int32 addresses + fp8 rows (mmap).
    Bit-exact vs MountPLE: same bytes, same scale, same dequant formula. No 48.7 GiB traffic.'''
    def __init__(self, directory):
        import json as _json
        d=Path(directory)
        meta=_json.loads((d/'compact.json').read_text())
        n=meta['address_count']
        self.addrs=torch.from_file(str(d/'addrs.u32'), shared=True, size=n, dtype=torch.int32)
        raw=torch.from_file(str(d/'rows.u8'), shared=True, size=n*meta['row_dim'], dtype=torch.uint8)
        self.rows=raw.view(torch.float8_e4m3fn).view(n, meta['row_dim'])
        self.scale=float(meta['scale']); self.meta=meta
        self.ple_revision=meta.get('ple_revision')
        print('compact PLE: %d rows, %.2f GiB mapped, scale=%g' % (n, (d/'rows.u8').stat().st_size/1024**3, self.scale))
    def lookup(self, indices):
        shape=indices.shape; flat=indices.detach().cpu().long().reshape(-1)
        assert bool((flat>=0).all()) and int(flat.max())<2**31
        f32=flat.to(torch.int32)
        pos=torch.searchsorted(self.addrs, f32)
        posc=pos.clamp(max=len(self.addrs)-1)
        assert bool((self.addrs[posc]==f32).all()), 'compact miss: address outside frozen prefix'
        out=self.rows.index_select(0, posc).to(torch.float32)*self.scale
        return out.reshape(*shape, self.rows.shape[-1]).flatten(-2)



print('stores ok; manifests:', [str(p) for p in find_ple_manifests()])
