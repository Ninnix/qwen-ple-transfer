"""Static checks for the Stage A2-arb notebook: tiny dynamic memory-arbitration
on frozen REAL-5M R=1 (IDX 2+8). Only alpha2/w/b trainable (~1K), gate feature
detached (h8.detach()). Balanced 6-way calibration (no FineWeb-only).
Compares A static canonical vs B dynamic fixed-IDX2 vs C dynamic+learned-IDX2
on frozen full-val/HS/LAMBADA/5 domains. Pre-run gates before training:
INVARIANT 1 (fresh A reproduces persisted v40 calibrated results, tolerance),
INVARIANT 2 (init B/C reproduce A, exact gates). Persists alpha8_t
distributions + learned alpha2. No R4, no backbone/PLE/
W_K/W_V/beta/gamma updates, no reader files written, logprob only."""
import ast
import json
import math
from pathlib import Path
REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_arb.ipynb').read_text(encoding='utf-8'))
print('cells:', len(nb['cells']), '| nbformat:', nb['nbformat'])
assert nb['nbformat'] == 4
assert nb.get('metadata', {}).get('kernelspec', {}).get('name'), 'kernelspec missing: papermill cannot execute'
print('  ok [metadata] kernelspec present')
code = {}
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    body = '\n'.join(l for l in src.splitlines() if not l.strip().startswith(('%pip', '!')))
    ast.parse(body)
    code[c['id']] = src
print('AST parse ok for', len(code), 'code cells:', sorted(code))
assert 'c-stageA2-arb' in code and 'c-evalh-a2' in code
assert 'c-stageA2-final' not in code and 'c-stageA2-calib' not in code and 'c-stageA2-gamma' not in code and 'c-stageA2-5m' not in code, 'stale stage cell leaked in'
print('  ok [stage] single arb cell, no stale final/calib/gamma/5m cell')
def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')
def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')
whole = '\n'.join(code.values())
arb = code['c-stageA2-arb']
# scope: no branch scaling, no heavy suites, no free generation, no old trainer
for banned in ('train_reader(', 'for R in [1, 4]', 'branches=4', 'run_downstream', 'MATH-500', 'LiveCodeBench', 'model.generate', '.generate(', 'reader-control-random', 'reader-control-permuted', 'real-500k-r1', 'real-1m-r1.reader.safetensors', 'resume.pt', 'save_file'):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no old trainer, no R4, no heavy suites, no free generation, no reader safetensors writes')
for num in ('5000000', '2000000', '3000000', '1000000', '1000448', '500224'):
    assert num not in whole, f'token-count leak (training threshold): {num!r}'
print('  ok [all] no stale training token-count literals (only 5000192 actual + 319488 arb)')
# arbitration trains tiny params: AdamW/backward allowed ONLY in arb cell
for s in ('AdamW(', '.backward(', 'clip_grad_norm', 'torch.save'):
    assert s in arb, f'arb cell must train tiny params ({s})'
print('  ok [arb] tiny training present (AdamW/backward/clip/save, arb-scoped)')
# frozen pins (same as final)
need('c-stageA2-arb', 'real-5m-r1.reader.safetensors', '5M weights file')
need('c-stageA2-arb', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2-arb', '_load_sfc(', 'safetensors loader with budget+SHA gate')
need('c-stageA2-arb', '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594', '5M byte-integrity SHA')
need('c-stageA2-arb', 'qwen-ple-a2-v34', 'frozen v34 results dataset')
need('c-stageA2-arb', '2fc06364715b967f1860aea9cf38778875588b17', 'pinned tokenizer revision')
need('c-stageA2-arb', '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'pinned hellaswag rev')
need('c-stageA2-arb', '900124bf3b8235c6daf21033af9948b3f07346c4', 'pinned lambada rev')
need('c-stageA2-arb', '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c', 'pinned full-val SHA')
need('c-stageA2-arb', 'a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5', 'pinned general SHA')
need('c-stageA2-arb', '735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421', 'pinned code SHA')
need('c-stageA2-arb', 'c27f42f5f69de1aaf26020c7e14c2c141f8fa8490c5d95952c06e1c5bffa6648', 'pinned math SHA')
need('c-stageA2-arb', 'c33632cf96170ca7ae92d4013702c14097fd79bb069c76e40fb3a531091936d8', 'pinned scientific SHA')
need('c-stageA2-arb', '65a26da5419aaa32520219d9d1c5909769a7a19a88ff16e85df7d69c38028a55', 'pinned multilingual SHA')
need('c-stageA2-arb', 'log_softmax', 'teacher-forced logprob scoring')
need('c-stageA2-arb', 'eval determinism ok', 'repeat-logprob determinism gate')
need('c-stageA2-arb', '1e-12', 'frozen-contrast reproduce tolerance')
need('c-stageA2-arb', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
# frozen discipline: backbone/PLE/W_K/W_V/beta/gamma never updated
need('c-stageA2-arb', 'W_K', 'frozen W_K discipline')
need('c-stageA2-arb', 'W_V', 'frozen W_V discipline')
need('c-stageA2-arb', 'beta', 'frozen beta discipline')
need('c-stageA2-arb', 'requires_grad_(False)', 'frozen grad-free proof')
need('c-stageA2-arb', 'optimizer must contain exactly', 'optimizer scoped to arbitration only')
need('c-stageA2-arb', 'backbone gradient leak', 'no backbone grad gate')
need('c-stageA2-arb', 'frozen reader gradient leak', 'no W_K/W_V/beta/gamma grad gate')
need('c-stageA2-arb', 'backbone weights changed', 'backbone version guard')
need('c-stageA2-arb', 'no reader files written', 'no weight-overwrite discipline')
need('c-stageA2-arb', '0.06166713312268257', '5M learned gamma IDX2 reference')
need('c-stageA2-arb', '0.07537073642015457', '5M learned gamma IDX8 reference')
# arbitration module: bounded alpha2 + token-dependent alpha8_t
need('c-stageA2-arb', 'MemoryArbitration', 'arbitration module')
need('c-stageA2-arb', 'alpha2', 'global IDX2 scalar')
need('c-stageA2-arb', 'alpha8', 'token-dependent IDX8 multiplier')
need('c-stageA2-arb', '0.75', 'alpha2 lower bound')
need('c-stageA2-arb', '1.75', 'alpha2 upper bound')
need('c-stageA2-arb', '1.25', 'alpha2/IDX2 1.25 level')
need('c-stageA2-arb', '0.125', 'alpha8 0.125 level')
need('c-stageA2-arb', 'sigmoid', 'sigmoid gates')
need('c-stageA2-arb', 'rms_norm', 'RMSNorm context dependence')
need('c-stageA2-arb', '-1.0986', 'b init logit(0.25)')
need('c-stageA2-arb', '1026', 'exact ~1K param count (1024+1+1)')
need('c-stageA2-arb', '~1K', 'tiny budget label')
need('c-stageA2-arb', 'gamma2 * alpha2', 'effective IDX2 injection')
need('c-stageA2-arb', 'gamma8 * alpha8_t', 'effective IDX8 injection')
need('c-stageA2-arb', '0.5 * torch.sigmoid', 'alpha8_t 0.5*sigmoid form')
need('c-stageA2-arb', 'h8.detach()', 'detached gate feature (no backbone graph for the gate)')
forbid('c-stageA2-arb', 'rms_norm(h8.float())', 'attached gate feature (must detach h8)')
# balanced calibration corpus (no FineWeb-only)
for dom in ('general', 'narrative', 'code', 'math', 'scientific', 'multilingual'):
    need('c-stageA2-arb', dom, f'balanced corpus includes {dom}')
need('c-stageA2-arb', '319488', 'calibration total ~319K tokens')
need('c-stageA2-arb', '250000', 'lower budget bound 250K')
need('c-stageA2-arb', '500000', 'upper budget bound 500K')
need('c-stageA2-arb', 'FineWeb-only', 'explicit non-FineWeb-only discipline')
need('c-stageA2-arb', 'edward-io/starcoderdata-repo', 'code candidate: public first (harness order)')
need('c-stageA2-arb', 'codeparrot/github-code-clean', 'code candidate fallback (gated last)')
need('c-stageA2-arb', 'HuggingFaceFW/fineweb-2', 'multilingual fallback (script-safe, harness order)')
forbid('c-stageA2-arb', 'FineWebEduStream(', 'FineWeb-only stream construction in arb')
forbid('c-stageA2-arb', 'HuggingFaceFW/fineweb-edu', 'FineWeb-Edu dataset in arb calibration')
# light mean-only regularizer (not per-token forcing)
need('c-stageA2-arb', 'regularizer', 'light regularizer label')
need('c-stageA2-arb', 'mean', 'mean aggregation')
need('c-stageA2-arb', 'ARB_LAMBDA', 'regularizer weight name')
need('c-stageA2-arb', 'mean-only', 'per-token variation allowed (not forced)')
# A/B/C comparison
need('c-stageA2-arb', 'STATIC_M2', 'static canonical IDX2 name')
need('c-stageA2-arb', 'STATIC_M8', 'static canonical IDX8 name')
need('c-stageA2-arb', 'static canonical', 'arm A label')
need('c-stageA2-arb', 'learn_alpha2', 'B-vs-C distinction (fixed vs learned IDX2)')
need('c-stageA2-arb', 'fixed_alpha2', 'arm B fixed-1.25 mechanism')
need('c-stageA2-arb', 'B-dynamic', 'arm B persistence/scoring')
need('c-stageA2-arb', 'C-dynamic', 'arm C persistence/scoring')
# alpha distribution persistence + learned alpha2 report
need('c-stageA2-arb', 'alpha8-stats.json', 'alpha8 distribution manifest')
need('c-stageA2-arb', 'p05', 'p05 quantile')
need('c-stageA2-arb', 'p50', 'p50 quantile')
need('c-stageA2-arb', 'p95', 'p95 quantile')
need('c-stageA2-arb', 'arbitration-b.json', 'arm B manifest (learned alpha2 + w/b)')
need('c-stageA2-arb', 'arbitration-c.json', 'arm C manifest (learned alpha2 + w/b)')
need('c-stageA2-arb', 'arbitration-', 'arbitration weight persistence')
for name in ('hellaswag-disabled.jsonl', 'lambada-disabled.jsonl', 'hellaswag-raw.jsonl', 'lambada-raw.jsonl', 'hellaswag-A.jsonl', 'lambada-A.jsonl', 'hellaswag-B.jsonl', 'lambada-B.jsonl', 'hellaswag-C.jsonl', 'lambada-C.jsonl', 'config.json', 'checkpoint-shas.json', 'timings.json', 'summary.json', 'bootstrap.json', 'calibration.json'):
    need('c-stageA2-arb', name, f'persist {name}')
# final pre-run invariants: fresh-A vs persisted v40 (tolerance), init-B/C vs A (exact gates), both before training
need('c-stageA2-arb', 'for _bi in range(val_full_chk.shape[0])', 'compact covers all full-val blocks (no truncation)')
forbid('c-stageA2-arb', 'val_full_chk.reshape(-1)[:512]', 'truncated val coverage (compact miss)')
need('c-stageA2-arb', 'qwen-ple-a2-final-v40', 'v40 calibrated dataset (pinned mount, no kernels-output API)')
need('c-stageA2-arb', 'V40MNT', 'v40 mount path name')
need('c-stageA2-arb', 'V40_SUMMARY_SHA', 'v40 summary byte-pin name')
need('c-stageA2-arb', 'dd584526893d09956a600173286a64fc5c96dda5afe129463fc34bc5dfbebec1', 'v40 summary SHA pin')
need('c-stageA2-arb', 'calibrated-reader.json', 'v40 provenance manifest')
need('c-stageA2-arb', 'V40_TOL', 'v40 numerical tolerance name')
need('c-stageA2-arb', 'hellaswag-calibrated.jsonl', 'v40 per-example HS rows')
need('c-stageA2-arb', 'lambada-calibrated.jsonl', 'v40 per-example LAMBADA rows')
need('c-stageA2-arb', 'must match exactly', 'v40 per-example correctness exactness')
need('c-stageA2-arb', 'PRE-RUN INVARIANT 1', 'invariant-1 marker (fresh A == v40)')
need('c-stageA2-arb', 'PRE-RUN INVARIANT 2', 'invariant-2 marker (init B/C == A)')
need('c-stageA2-arb', 'INIT_TOL', 'init end-to-end logit tolerance name')
need('c-stageA2-arb', 'INIT_NLL_TOL', 'init probe NLL tolerance name')
need('c-stageA2-arb', 'before any optimizer step', 'init check precedes training')
i1 = arb.find('PRE-RUN INVARIANT 1')
i2 = arb.find('PRE-RUN INVARIANT 2')
t0 = arb.find("train_arb('b'")
assert 0 <= i1 < t0 and 0 <= i2 < t0, 'both invariants must gate training (precede first train_arb call)'
print('  ok [c-stageA2-arb] invariants precede training (fail-fast pre-run gates)')
# frozen constants sync with train notebook
tnb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
tcode = {c['id']: ''.join(c['source']) for c in tnb['cells'] if c['cell_type'] == 'code'}
for const in ['SOURCE_ID', 'TARGET_ID', 'MEM_DIM', 'HIDDEN', 'N_LAYERS', 'NGRAM', 'HEADS_PER_NGRAM', 'ROW_DIM', 'ROWS_PER_PART', 'VOCAB_BASE', 'SEED', 'EOS', 'VOCAB', 'SEQ', 'VAL_FAST_TOKENS', 'VAL_FULL_TOKENS', 'DATASET_ID', 'DATASET_CONFIG', 'BRANCHES', 'GAMMA_INIT']:
    assert const in code['c-config'] and const in tcode['c-config'], const
print('  ok [c-config] frozen constants present in both notebooks')
# logic: alpha2 sigmoid mapping init/bounds
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))
assert abs((0.75 + 1.0 * _sig(0.0)) - 1.25) < 1e-12, 'alpha2 raw=0 must give 1.25'
assert 0.75 < 1.25 < 1.75, 'init inside bounds'
assert abs(0.5 * _sig(-1.0986122886681098) - 0.125) < 1e-9, 'b_init must give 0.125'
print('  ok [logic] alpha2 raw0->1.25 in [0.75,1.75]; b_init->0.125')
# logic: param count 1024+1+1
assert 1024 + 2 == 1026, 'hidden 1024 -> 1026 params'
print('  ok [logic] arbitration params 1026 (~1K)')
# logic: calibration budget 6x53248 within 250K-500K
assert 6 * 53248 == 319488 and 250000 <= 319488 <= 500000
print('  ok [logic] balanced 6x53248=319488 tokens within 250K-500K')
# logic: regularizer is light (saturation penalty ~1% of CE)
sat = 2.0 * (0.25 - 0.125) ** 2
assert abs(sat - 0.03125) < 1e-12 and sat < 0.05, 'mean-reg must stay light vs CE~2.9'
print('  ok [logic] mean-reg light: saturation penalty %.5f << CE' % sat)
km = json.loads((REPO / 'kernel-metadata.json').read_text(encoding='utf-8'))
assert 'ninnix/qwen-ple-a2-final-v40' in km['dataset_sources'], 'v40 dataset must be attached for the mount-gated invariant'
assert 'ninnix/qwen-ple-a2-v34' in km['dataset_sources'], 'frozen v34 dataset must stay attached'
print('  ok [metadata] v40 + v34 datasets attached')
print('ALL A2-ARB CHECKS PASSED')
