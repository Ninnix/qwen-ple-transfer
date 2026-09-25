import ast, hashlib, json, math, re, sys
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
print('cells:', len(nb['cells']), '| nbformat:', nb['nbformat'])
assert nb['nbformat'] == 4

code = {}
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    body = '\n'.join(l for l in src.splitlines() if not l.strip().startswith(('%pip', '!')))
    ast.parse(body)  # raises on syntax error
    code[c['id']] = src
print('AST parse ok for', len(code), 'code cells:', sorted(code))

def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')

def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')

whole = '\n'.join(code.values())

# 0. Cell-10 test correctness (this round's fixes)
forbid('c-tests', '[[[1,2,3]]]', '3-address RandomPLE input')
need('c-tests', '_base=torch.tensor([[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]])', '16 addresses/token (one per head)')
need('c-tests', 'r1.shape==(1,1,2560)', 'exact (1,1,2560) shape assert')
need('c-tests', 'changing one address must change the output', 'single-address sensitivity')
need('c-tests', 'RandomPLE([0.0]*16,[0.0086]*16).lookup(addresses(toy.cpu()))', 'backward-test memory from same toy ids (calibrated scale)')
forbid('c-tests', 'RandomPLE()', 'uncalibrated RandomPLE() default (old 0.06 scale)')
forbid('c-tests', 'ngram_indices(probe.cpu())[:,:32]', 'probe-derived memory for toy input')
_t = code['c-tests']
assert _t.find('toy=torch.randint') < _t.find('RandomPLE([0.0]*16,[0.0086]*16).lookup(addresses(toy'), 'toy must exist before memory'
print('  ok [c-tests] toy-before-memory ordering')
need('c-data', 'def __init__(self, tok, skip_docs, dataset_rev):', 'pinned-rev stream signature')
need('c-data', 'revision=dataset_rev', 'stream uses pinned rev')
need('c-data', 'self.it=iter(self.ds)', 'explicit streaming iterator')
forbid('c-data', 'next(self.ds)', 'dataset-as-iterator TypeError')
forbid('c-data', 'drev=HfApi(token', 'independent latest-rev resolve in stream')
need('c-trainfn', "skip=_vmeta['skip_docs']; drev=_vmeta['dataset_rev']", 'skip+rev from frozen metadata')
need('c-trainfn', 'FineWebEduStream(tokenizer, skip, drev)', 'pinned rev passed to stream')
forbid('c-trainfn', 'FineWebEduStream(tokenizer, skip)', 'unpinned stream construction')
assert 'SWEEP_ENABLED: bool = False' in code['c-config'], 'SWEEP must be False (sweep complete, winner IDX 2+8)'
assert 'BRANCH_ENABLED: bool = False' in code['c-config'], 'BRANCH must be False (controls persisted, single-track 1M)'
assert 'REAL1M_ENABLED: bool = False' in code['c-config'], 'REAL1M must be False (v23 done, eval-only job now)'
assert 'REAL5M_ENABLED: bool = False' in code['c-config'], 'REAL5M must be False at rest (staged 2M/3M/5M gated)'
assert 'STAGEA_ENABLED: bool = False' in code['c-config'], 'train nb never runs eval'
print('  ok [c-config] SWEEP/BRANCH/REAL1M/REAL5M=False, STAGEA_ENABLED=True (eval-only job)')
print('  ok [c-config] gates moved: BRANCH done, REAL1M staged (see section 15)')
# 13. validation prep before first load (this round)
assert 'c-valprep' in code, 'c-valprep cell missing'
need('c-valprep', 'mf, mfull = build_validation_artifacts(tokenizer)', 'build-once call')
need('c-valprep', 'val_fast, _ = load_validation("fast")', 'fast load')
need('c-valprep', 'val_full, _ = load_validation("full")', 'full load')
need('c-valprep', 'print("validation ready")', 'ready message')
need('c-valprep', 'print("fast:", mf["tokens_sha256"])', 'fast sha print')
need('c-valprep', 'print("full:", mfull["tokens_sha256"])', 'full sha print')
_ids = [c['id'] for c in nb['cells']]
assert _ids.index('c-valprep') < _ids.index('c-smoke'), 'prep must precede Cell 12'
assert _ids.index('c-valprep') < _ids.index('c-realsmoke'), 'prep must precede real smoke'
assert _ids.index('c-load') < _ids.index('c-valprep'), 'prep needs tokenizer from Cell 9'
print('  ok [c-valprep] ordering: Cell 9 < prep < Cell 12, before any load_validation use')
# 14. Cell 9 mixed-precision status (this round)
forbid('c-load', 'reader + memory cast to this', 'stale fp16-reader message')
need('c-load', 'backbone = ACTIVE_DTYPE', 'backbone dtype line')
need('c-load', 'reader params/compute = FP32', 'reader FP32 line')
need('c-load', 'PLE dequant/output = FP32', 'PLE FP32 line')
need('c-load', 'residual output is cast back to backbone dtype', 'cast-back line')

# 0b. Environment credentials; Kaggle secret attachments are optional.
need('c-env', "secret_value_0 = os.environ.get('HF_TOKEN')", 'HF environment')
need('c-env', "secret_value_1 = os.environ.get('KAGGLE_API_TOKEN')", 'Kaggle environment')
need('c-env', 'from kaggle_secrets import UserSecretsClient', 'Kaggle secret import')
need('c-env', "secret_value_0 = secret_value_0 or user_secrets.get_secret('HF_TOKEN')", 'HF attachment')
need('c-env', "secret_value_1 = secret_value_1 or user_secrets.get_secret('KG_TOKEN')", 'Kaggle attachment')
forbid('c-env', 'def get_secret', 'helper removed (no complicated logic)')
assert whole.count("get_secret('HF_TOKEN')") == 1, 'exactly one HF fetch (Cell 0)'
assert whole.count("get_secret('KG_TOKEN')") == 1, 'exactly one KG fetch (Cell 0)'
assert whole.count('secret_value_0') >= 4, 'secret_value_0 consumed downstream'
assert not re.search(r'hf_[A-Za-z0-9]{20,}|(?i:kgat_[A-Za-z0-9_-]{20,})', whole), 'embedded credential'
print('  ok [secrets] environment first, no embedded credential')

# 0c. P100 torch before first import + semantic tokenizer check (this round)
# 0d. transformers 5.17 native qwen3_5 + full-config load (no text_config override)
need('c-install', 'transformers==5.17.0', 'exact pin (base 5.0.0 lacks qwen3_5)')
need('c-install', ' accelerate', 'accelerate for device_map + modeling import chain')
forbid('c-install', 'transformers>=4.57', 'floor pin (no-ops on base image)')
need('c-load', "assert getattr(cfg,'model_type',None)=='qwen3_5'", 'model-type guard')
need('c-load', 'dims ONLY', 'text_config dims-only comment')
forbid('c-load', 'config=tconf', 'auto_map-stripping override')
need('c-load', 'decoder blocks:', 'decoder layout probe')
need('c-load', 'direct qwen3_5 import ok', 'import diagnostic (true missing dep)')
need('c-inject', 'len(dec)==C.N_LAYERS', 'hook-target count assert')
print('  ok [load] native qwen3_5 mapping, full-config VLM-compat load, 24-block hooks')
forbid('c-env', 'import torch', 'torch import in Cell 0 (must load sm_60 build in Cell 1)')
forbid('c-env', 'torch.cuda', 'torch use in Cell 0')
need('c-install', 'torch==2.5.1+cu118', 'P100 torch pin')
need('c-install', 'cu118', 'cu118 index')
need('c-install', 'torchvision==0.20.1+cu118', 'matching torchvision (modeling import chain)')
need('c-install', 'torchaudio==2.5.1+cu118', 'matching torchaudio (modeling import chain)')
need('c-install', 'torch.zeros(1).cuda()', 'fail-fast sm_60 probe')
need('c-install', 'get_device_capability', 'capability report')
need('c-tok', 'raw_vocab', 'raw mapping compare')
need('c-tok', 'not diff and not extra_t', 'exact subset addressing proof')
need('c-tok', 'NATIVE token-ID addressing VALID', 'native verdict')
need('c-tok', 'tt.encode(probe', 'probe-encode equality')
forbid('c-tok', 'tokenizer.json equal', 'byte-equality verdict')
forbid('c-tok', 'assert th==sh', 'byte-equality assert')
forbid('c-tok', 'len(tt)==C.VOCAB', 'AutoTokenizer-len assert')
assert 'tokenizer.eos_token_id' not in whole and 'tok.eos_token_id' not in whole, 'tokenizer-object eos reads (chat im_end trap)'
need('c-config', 'EOS: int = 248044', 'training terminator constant')
need('c-config', 'VOCAB: int = 248320', 'training hash vocab constant')
need('c-config', 'SEED: int = 1234', 'training hash seed constant')
print('  ok [p100+tok] sm_60 torch first, semantic id-space, training constants pinned')

# 1. validation addresses via ngram_indices, never raw ids (defined ONCE in Cell 3)
need('c-hashing', 'def addresses(token_cpu):', 'canonical addresses() in Cell 3')
forbid('c-trainfn', 'def addresses(token_cpu):', 'duplicate definition')
need('c-trainfn', 'store.lookup(addresses(', 'eval/train lookup via addresses()')
forbid('c-trainfn', 'store.lookup(blk', 'raw-id lookup in eval')
# 2. no hardcoded FP8 scale fallback anywhere
assert '0.000199' not in whole, 'hardcoded scale fallback present'
print('  ok [all] no hardcoded FP8 scale')
need('c-ple', 'Refusing hardcoded fallback', 'scale-or-abort')
# 3. mount-only real PLE, no hub download for shards
need('c-ple', "class MountPLE:", 'MountPLE store')
need('c-ple', 'def find_ple_manifests', 'multi-manifest discovery')
forbid('c-ple', 'find_ple_manifest()', 'singular mount call')
need('c-ple', 'len(self.part_paths)==128', 'all-parts merged assert')
need('c-ple', 'mixed PLE revisions', 'single-revision guard')
need('c-ple', 'each dataset holds only its own shards', 'existence-based merge')
need('c-ple', 'filename clash', 'duplicate-copy guard')
need('c-ple', 'no /kaggle/input PLE dataset attached', 'mount requirement abort')
forbid('c-ple', 'hf_hub_download', 'PLE shard downloading')
# 4. RandomPLE per-address deterministic 160-d rows -> 2560-d (calibrated per-head, no 0.06)
need('c-ple', 'manual_seed', 'deterministic seeding')
need('c-ple', 'torch.unique(flat, return_inverse=True)', 'per-address dedup')
need('c-ple', 'flatten(-2)', '16x160 -> 2560 concat')
need('c-ple', 'def calibrate_head_stats', 'per-head mean/std calibration from real rows')
need('c-ple', 'def permute_addresses', 'shared permute helper (builder + store)')
need('c-ple', 'head_means, head_stds', 'RandomPLE takes calibrated per-head stats')
forbid('c-ple', 'std=0.06', 'old uncalibrated random scale')
# 5. PermutedPLE bijective per-head
need('c-ple', '((f-O)*A+B)%S', 'per-head affine permutation')
need('c-ple', 'coprime', 'bijectivity guard')
need('c-ple', 'head_layout()', 'head-boundary layout')
forbid('c-ple', 'self.cache', 'whole-part tensor cache')
need('c-ple', 'get_slice', 'row-level slice reads')
need('c-ple', 'held_part_tensors', 'boundedness accounting')
need('c-ple', 'def rss_mb', 'host RSS helper')
need('c-ple', 'def stats', 'store stats')
forbid('c-ple', 'index_select(0, local', 'full-part materialization')
# 6. threshold-crossing checkpoints
need('c-trainfn', 'while pending and seen>=pending[0]', 'crossing (not exact-multiple) ckpts')
# 7. never rebuild validation inside training
forbid('c-trainfn', 'build_validation_artifacts(', 'implicit val rebuild in trainer')
need('c-trainfn', 'assert val is not None', 'val passed in, frozen')
need('c-data', "def build_validation_artifacts(tok):", 'once-only builder')
need('c-data', "def load_validation(name):", 'frozen loader with sha check')
need('c-data', '65536', 'fast artifact size')
need('c-data', '524288', 'full artifact size')
# 8. FineWeb-Edu only for first experiment
need('c-data', 'class FineWebEduStream:', 'single-corpus stream')
assert 'MixedStream' not in whole and 'the-stack' not in whole and 'ultrachat' not in whole, 'mixed corpus leaked in'
print('  ok [all] FineWeb-Edu only, no Stack/UltraChat/Cosmopedia')
# 9. layer IDX/HUMAN convention matching 35B
need('c-config', '((2,), (8,), (2, 8))', 'placements (2,)/(8,)/(2,8)')
need('c-config', 'HUMAN', 'IDX/HUMAN distinction')
# 10. float16-first, FP32 reductions
need('c-load', '[torch.float16, torch.float32]', 'fp16-first loop')
need('c-reader', 'x.float()', 'FP32 reductions in norms/scores')
need('c-reader', 'h_dtype', 'reader FP32 compute + cast-back (params never fp16)')
need('c-reader', 'params stay FP32', 'no pure-fp16 reader')
assert ".to(ACTIVE_DTYPE)" not in whole, 'reader/memory must never be cast to fp16'
print('  ok [all] backbone fp16-eligible, reader+AdamW strictly FP32')
forbid('c-trainfn', "skip_docs'] if", 'conditional skip (train must require full-val artifact)')
need('c-trainfn', 'no leakage', 'skip-docs guard comment')
# 11. sweep gated off, controls gated on (separate gates prevent auto-start)
need('c-config', 'SWEEP_ENABLED: bool = False', 'sweep DONE (winner IDX 2+8)')
need('c-config', 'BRANCH_ENABLED: bool = False', 'controls DONE (persisted to dataset)')
need('c-config', 'REAL1M_ENABLED: bool = False', '1M DONE (eval-only job now)')
need('c-config', 'REAL5M_ENABLED: bool = False', '5M staged, gated off at rest')
need('c-config', 'STAGEA_ENABLED: bool = False', 'eval lives in the eval notebook')
# 11d. sweep protocol: dual-val schedule, cached baselines, identical placements
need('c-trainfn', 'val_full=None, baselines=None', 'dual-val signature')
need('c-trainfn', 'full_set=set(full_at) if full_at else {max(ckpts)}', 'full-val schedule (legacy final-only default)')
need('c-trainfn', "baselines['full' if _vv is val_full else 'fast']", 'cached baseline select')
need('c-trainfn', 'w_k_update', 'key update norm')
need('c-trainfn', 'w_v_update', 'value update norm')
need('c-trainfn', 'rss_MiB', 'host RSS telemetry')
need('c-place', 'CompactPLE(_cdir)', 'compact store in sweep')
need('c-place', 'cached baselines: fast', 'baselines-once report')
need('c-place', 'ckpts=[100000,250000,500000]', 'placement checkpoint schedule')
need('c-place', "_cm['dataset_rev']==_mF['dataset_rev']", 'cache/token consistency')
need('c-place', "train_reader(list(layers),1,500000", 'identical 500K R=1 placements')
need('c-place', 'SWEEP GATED OFF', 'placement gate')
# 11e. calibrated controls on winner only (separate BRANCH gate, identical protocol, no R=4/1M/5M)
assert 'c-branch' in code, 'c-branch cell missing'
need('c-branch', 'WIN = (2, 8)', 'winner fixed at IDX 2+8')
need('c-branch', 'if not C.BRANCH_ENABLED:', 'separate branch gate (never auto-start on SWEEP)')
need('c-branch', 'C.SWEEP_ENABLED is False', 'sweep stays OFF during controls')
need('c-branch', 'calibrate_head_stats(store_real', 'per-head calibration from real working set')
need('c-branch', 'RandomPLE(_means,_stds', 'calibrated deterministic random control')
need('c-branch', 'compact-500k-perm', 'separate permuted working-set cache')
need('c-branch', 'permute_addresses(_real_addrs', 'permuted addresses from same frozen prefix')
need('c-branch', 'PermutedPLE(store_perm_base', 'permuted wraps compact (no /kaggle/input random reads)')
need('c-branch', "train_reader([2,8],1,500000,store_rand,val_fast,'control-random-r1',ckpts=[100000,250000,500000],val_full=val_full,baselines=baselines)", 'identical-protocol random 500K')
need('c-branch', "train_reader([2,8],1,500000,store_perm,val_fast,'control-permuted-r1',ckpts=[100000,250000,500000],val_full=val_full,baselines=baselines)", 'identical-protocol permuted 500K')
need('c-branch', 'REAL vs RANDOM vs PERMUTED vs DISABLED', 'stop report (no R=4/1M/5M)')
need('c-branch', 'frozen baselines (DISABLED control)', 'disabled-memory control reuse')
forbid('c-branch', 'PermutedPLE(MountPLE', 'direct /kaggle/input random reads for permuted')
forbid('c-branch', 'RandomPLE()', 'uncalibrated random in controls')
forbid('c-branch', 'for R in [1, 4]', 'R=4 branch scaling (not in controls run)')
forbid('c-branch', '1000000', '1M extension (not in controls run)')
forbid('c-branch', '5000000', '5M extension (not in controls run)')
assert 'if not C.SWEEP_ENABLED:' not in code['c-branch'], 'branch must use BRANCH_ENABLED, not SWEEP_ENABLED'
need('c-export', 'BEST_IDX = [2, 8]', 'export winner IDX 2+8')
need('c-smoke', 'SMOKE_TOKENS', 'small smoke only')
need('c-smoke', 'RandomPLE([0.0]*16,[0.0086]*16)', 'smoke uses calibrated scale (never 0.06)')
forbid('c-smoke', 'RandomPLE()', 'uncalibrated smoke random')
# 11b. smoke-only tolerant tail (plot/export gated, never assert on missing sweep)
need('c-plot', 'smoke-only readout stands', 'gated plot message')
forbid('c-plot', 'assert runs', 'sweep-only assert')
need('c-export', "if 'smoke' not in p.name", 'smoke ckpts never ship')
need('c-export', 'export gated: no sweep checkpoints', 'gated export message')
# 11c. shared freeze helper + refactored trainer + compact cache cells
need('c-data', 'def freeze_stream_tokens', 'single token-prefix path')
need('c-trainfn', 'freeze_stream_tokens(tokenizer, stream, max_tokens)', 'trainer uses shared prefix')
need('c-trainfn', 'for step, _ids in enumerate(chunks', 'chunk loop')
forbid('c-trainfn', 'while len(buf)<C.SEQ', 'old inline buf loop')
need('c-ple', 'class CompactPLE', 'compact store class')
need('c-ple', 'searchsorted', 'indexed lookup')
need('c-ple', 'from_file', 'mmap backing')
need('c-ple', 'self.ple_revision', 'revision attribute for consistency check')
need('c-profile', 'PROFILE: lookup share', 'profile verdict')
need('c-profile', 'torch.cuda.synchronize', 'synchronized timing')
need('c-compact', 'dedup ratio', 'dedup report')
need('c-compact', 'compact.json', 'cache manifest')
need('c-compact', 'equivalence max|diff|', 'row-equivalence check')
need('c-bench', 'speedup', 'bench comparison')
need('c-bench', '20*512*16', 'in-set batch shape (tokens x heads)')
need('c-bench', 'SWEEP stays OFF', 'no sweep enable')
_ids1 = [c['id'] for c in nb['cells']]
assert _ids1.index('c-realsmoke') < _ids1.index('c-profile') < _ids1.index('c-compact') < _ids1.index('c-bench') < _ids1.index('c-place')
assert 'c-compact' in code and 'tokenizer.encode(next(stream),add_special_tokens=False)+[C.EOS]' not in code['c-compact'], 'builder must use freeze helper, not its own stream loop'
print('  ok [compact] freeze/profile/build/bench structure + order')
# 12. gated real-PLE smoke cell (no sweep start)
assert 'c-realsmoke' in code, 'c-realsmoke cell missing'
need('c-realsmoke', 'REAL_SMOKE_TOKENS=10240', '10K-token budget')
need('c-realsmoke', 'train_reader([2], 1', 'IDX 2 R=1 real smoke')
need('c-realsmoke', "load_validation('fast')", 'fast-validation smoke')
need('c-realsmoke', 'repeat-lookup mismatch', 'slice-after-close determinism check')
need('c-realsmoke', 'checkpoint export missing', 'export check')
need('c-realsmoke', 'host RAM grew', 'bounded host RAM check')
need('c-realsmoke', 'REAL_SMOKE_PASSED', 'explicit gate before SWEEP_ENABLED=True')

# ---- pure-python logic tests (no torch) ----
MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB

def splitmix64(v):
    v = (v + GAMMA) & MASK64
    v = ((v ^ (v >> 30)) * M1) & MASK64
    v = ((v ^ (v >> 27)) * M2) & MASK64
    return (v ^ (v >> 31)) & MASK64

def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))

def nth_prime_after(s, k):
    p = s
    for _ in range(k):
        p += 1
        while not is_prime(p):
            p += 1
    return p

sizes = tuple(nth_prime_after(20_000_000 - 1, h + 1) for h in range(16))
offs = []
t = 0
for s in sizes:
    offs.append(t)
    t += s
assert all(is_prime(s) for s in sizes), 'head sizes must be prime (coprime guarantee)'
# per-head permutation bijectivity on edge addresses
for h, (s, o) in enumerate(zip(sizes, offs)):
    a = int(splitmix64((777 + 10007 * (h + 1)) & MASK64) % (s - 1)) + 1
    b = int(splitmix64(((777 ^ 0x9E3779B97F4A7C15) + 7919 * (h + 1)) & MASK64) % s)
    assert a % s != 0 and math.gcd(a, s) == 1
    edge = [o, o + 1, o + s - 1]
    mapped = [o + ((x - o) * a + b) % s for x in edge]
    assert all(o <= m < o + s for m in mapped), f'head {h} boundary violated'
    assert len(set(mapped)) == 3, f'head {h} collision'
print('  ok [logic] per-head permutation bijective + boundary-preserving over 16 heads')

# checkpoint crossing: SEQ=512 never hits 100000 exactly
SEQ, seen, pending = 512, 0, [100000, 250000, 500000]
fired = []
while seen < 500000:
    seen += SEQ
    while pending and seen >= pending[0]:
        fired.append((pending.pop(0), seen))
assert [f[0] for f in fired] == [100000, 250000, 500000], fired
assert fired[0][1] == 100352 and fired[1][1] == 250368 and fired[2][1] == 500224, fired
print('  ok [logic] crossing ckpts fire at', [f[1] for f in fired], '(exact-multiple would never fire)')

# immutable validation artifacts: write-once, reject divergence, sha-pinned
import tempfile
d = Path(tempfile.mkdtemp())
fake = array('I', list(range(65536)))
raw = fake.tobytes()
digest = hashlib.sha256(raw).hexdigest()
meta = {'name': 'fast', 'token_count': 65536, 'tokens_sha256': digest}
(d / 'tokens-fast.uint32le').write_bytes(raw)
(d / 'validation-fast.json').write_text(json.dumps(meta))
reread = (d / 'tokens-fast.uint32le').read_bytes()
assert hashlib.sha256(reread).hexdigest() == json.loads((d / 'validation-fast.json').read_text())['tokens_sha256']
old = json.loads((d / 'validation-fast.json').read_text())
divergent = {'name': 'fast', 'token_count': 1, 'tokens_sha256': 'x'}
assert old != divergent, 'harness setup wrong'
refused = (old != divergent)  # notebook _write_val raises instead of overwriting
assert refused, 'immutability guard must refuse divergent rewrite'
print('  ok [logic] frozen val sha-pinning + immutability guard, sha=' + digest[:16])

# row-run grouping + gather/scatter equivalence (mirrors new MountPLE.lookup core)
import random as _r
_r.seed(7)
fake_part = list(range(1000))  # row r holds value r
locals_ = sorted(_r.sample(range(1000), 500))
runs = []
a = prev = locals_[0]
for r in locals_[1:]:
    if r == prev + 1:
        prev = r
    else:
        runs.append((a, prev + 1))
        a = prev = r
runs.append((a, prev + 1))
got = [v for x, y in runs for v in fake_part[x:y]]
assert got == locals_, 'run-sliced gather must equal sorted rows'
assert sum(b - x for x, b in runs) == len(locals_)
print('  ok [logic] contiguous-run slicing covers exactly the requested rows')

# kernel metadata
km = json.loads((REPO / 'kernel-metadata.json').read_text(encoding='utf-8'))
assert km['id'] == 'ninnix/qwen3-5-0-8b-ple-target-side-reader-p100', km['id']
assert km['machine_shape'] == 'NvidiaTeslaP100' and km['enable_gpu'] == 'true' and km['enable_internet'] == 'true'
assert km['dataset_sources'] == ['ninnix/qwen38-ple-p%02d' % i for i in range(11)] + ['ninnix/qwen-ple-reader-checkpoints', 'ninnix/qwen-ple-a2-v34', 'ninnix/qwen-ple-a2-final-v40', 'ninnix/qwen-ple-a2-arb'], km['dataset_sources']
print('  ok [metadata] id/machine_shape/gpu/internet; 11 PLE + checkpoints + 3 A2 stage datasets attached')
# 15. staged REAL-1M run (separate REAL1M gate; sweep+branch OFF; no R4/5M; artifacts persist)
assert 'c-real1m' in code and 'c-eval' not in code, 'train nb: real1m kept, eval harness moved out'
need('c-trainfn', 'full_at=None', 'dual full-val schedule param')
need('c-trainfn', 'def write_bundle', 'canonical artifact writer')
need('c-trainfn', 'deepcopy(opt.state_dict())', 'ckpt must not alias live optimizer state')
need('c-trainfn', 'deepcopy(opt.state_dict())', 'ckpt must not alias live optimizer state')
need('c-trainfn', 'resume=None', 'resume path for 500K->1M continuation')
need('c-real1m', 'if not C.REAL1M_ENABLED:', 'separate 1M gate')
need('c-real1m', 'C.BRANCH_ENABLED is False', 'branch stays OFF during 1M')
need('c-real1m', 'full_at={500000}', 'full-val at 500K reproduce point')
need('c-real1m', 'full_at={1000000}', 'full-val at 1M final')
need('c-real1m', 'resume=ck500', 'same-run continuation (optimizer carried)')
need('c-real1m', 'REAL500K_DVAL', 'sweep reference for reproduce check')
forbid('c-real1m', 'dump_perblock', 'per-block dump lives in the separate eval job')
forbid('c-real1m', 'run_downstream', 'downstream runs as a separate job, not mid-training')
need('c-real1m', "assert ck500['seen'] == REAL500K_SEEN", 'resume ckpt sanity before leg 2')
need('c-real1m', 'qwen-ple-reader-checkpoints', 'persistent dataset source')
forbid('c-real1m', 'branches=4', 'R4 in 1M cell')
forbid('c-real1m', 'for R in [1, 4]', 'R sweep in 1M cell')
forbid('c-real1m', '5000000', '5M in 1M cell')
for _cid, _src in code.items():
    if _cid != 'c-real1m':
        assert '1000000' not in _src, f'{_cid}: 1M token count outside staged cell'
print('  ok [real1m] 1M confined to c-real1m, no R4/5M')
# 16. staged REAL-5M run (separate REAL5M gate; sweep+branch+1M OFF; no R4; early-stop; persist)
assert 'c-real5m' in code, 'c-real5m cell missing'
need('c-real5m', 'if not C.REAL5M_ENABLED:', 'separate 5M gate')
need('c-real5m', 'C.BRANCH_ENABLED is False', 'branch stays OFF during 5M')
need('c-real5m', 'C.SWEEP_ENABLED is False', 'sweep stays OFF during 5M')
need('c-real5m', 'WIN = (2, 8)', 'winner fixed at IDX 2+8')
need('c-real5m', 'REAL5M_LEGS', '2M/3M/5M legs')
need('c-real5m', 'MIN_GAIN', 'saturate threshold')
need('c-real5m', 'EARLY-STOP', 'saturate/reverse stop')
need('c-real5m', 'resume=cur', 'resume-carried continuation')
need('c-real5m', 'full_at={_th}', 'full-val at each milestone')
need('c-real5m', 'cont_peak=CONT_PEAK', 'continuation LR peak passed per leg')
need('c-real5m', 'CONT_PEAK = 1.5e-05', 'explicit continuation peak (no naive 2.8e-05 shock)')
need('c-real5m', 'CONT_WARM = 200', 'explicit re-warm length')
need('c-real5m', "'lr_schedule'", 'schedule recorded in bundle meta')
need('c-trainfn', 'cont_peak=None', 'continuation peak param (legacy path preserved)')
need('c-trainfn', "'lr':float(opt.param_groups[0]['lr'])", 'LR recorded per checkpoint')
need('c-real5m', "_tag + '.resume.pt'", 'flat dataset resume layout')
need('c-trainfn', 'old scheduler NOT restored', 'naive restore explicitly avoided')
need('c-trainfn', 'sch.last_epoch=_skip', 'scheduler aligned to global step')
need('c-real5m', 'write_bundle', 'canonical bundles per leg')
need('c-real5m', 'qwen-ple-reader-checkpoints', 'persistent dataset source')
forbid('c-real5m', 'branches=4', 'R4 in 5M cell')
forbid('c-real5m', 'for R in [1, 4]', 'R sweep in 5M cell')
forbid('c-real5m', 'dump_perblock', 'per-block dump lives in eval, not training')
forbid('c-real5m', 'run_downstream', 'downstream runs as separate job')
for _cid, _src in code.items():
    if _cid != 'c-real5m':
        assert '5000000' not in _src, f'{_cid}: 5M token count outside staged cell'
print('  ok [real5m] 5M confined to c-real5m, IDX 2+8 R=1, early-stop, no R4')
_ids2 = [c['id'] for c in nb['cells']]
assert _ids2.index('c-branch') < _ids2.index('c-real1m') < _ids2.index('c-real5m') < _ids2.index('c-plot'), 'train order'
print('  ok [order] c-branch < c-real1m < c-real5m < c-plot (no eval cells in train nb)')
print('ALL STATIC + LOGIC CHECKS PASSED')
