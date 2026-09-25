"""Static checks for the Stage A2-final notebook: runtime-calibrated REAL-5M
(1.25, 0.125) vs RAW vs DISABLED. Inference-only, frozen v34 HS/LAMBADA +
full-val + 5 domains, raw weights unchanged, multipliers in manifest only,
no grid rerun, no training, no R4, logprob only."""
import ast
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_final.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA2-final' in code and 'c-evalh-a2' in code
assert 'c-stageA2-calib' not in code and 'c-stageA2-gamma' not in code, 'stale stage cell leaked in'
print('  ok [stage] single final cell, no stale calib/gamma cell')
def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')
whole = '\n'.join(code.values())
for banned in ('train_reader(', 'AdamW(', 'from torch.optim', '.backward(', 'clip_grad_norm', 'freeze_stream_tokens',
               'FineWebEduStream', 'SWEEP_ENABLED', 'BRANCH_ENABLED', 'REAL1M_ENABLED', 'REAL5M_ENABLED',
               'STAGEA_ENABLED', 'torch.save', 'for R in [1, 4]', "'optimizer':",
               'run_downstream', 'MATH-500', 'LiveCodeBench',
               'model.generate', '.generate(',
               'reader-control-random', 'reader-control-permuted', 'real-500k-r1', 'real-1m-r1.reader.safetensors',
               'resume.pt', 'branches=4'):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no training, no other-arm reruns, no R4, no heavy suites, no free generation')
for num in ('5000000', '2000000', '3000000', '1000000', '1000448', '500224'):
    assert num not in whole, f'token-count leak (training threshold): {num!r}'
print('  ok [all] no training token-count literals (only 5000192 actual)')
for grid in ('IDX2_MULTS', 'IDX8_MULTS', '_pareto_keys(', 'grid.json', 'pareto.json', 'GAMMA_MULTS', 'ABLATION'):
    assert grid not in whole, f'grid rerun leak (final must be 3 arms only): {grid!r}'
print('  ok [all] no calibration-grid rerun (focused 3-arm job only)')
need('c-stageA2-final', 'real-5m-r1.reader.safetensors', '5M weights file')
need('c-stageA2-final', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2-final', '_load_sfc(', 'safetensors loader with budget+SHA gate')
need('c-stageA2-final', '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594', '5M byte-integrity SHA')
need('c-stageA2-final', 'qwen-ple-a2-v34', 'frozen v34 results dataset')
need('c-stageA2-final', '2fc06364715b967f1860aea9cf38778875588b17', 'pinned tokenizer revision')
need('c-stageA2-final', '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'pinned hellaswag rev')
need('c-stageA2-final', '900124bf3b8235c6daf21033af9948b3f07346c4', 'pinned lambada rev')
need('c-stageA2-final', '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c', 'pinned full-val SHA')
need('c-stageA2-final', 'a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5', 'pinned general SHA')
need('c-stageA2-final', '735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421', 'pinned code SHA')
need('c-stageA2-final', 'c27f42f5f69de1aaf26020c7e14c2c141f8fa8490c5d95952c06e1c5bffa6648', 'pinned math SHA')
need('c-stageA2-final', 'c33632cf96170ca7ae92d4013702c14097fd79bb069c76e40fb3a531091936d8', 'pinned scientific SHA')
need('c-stageA2-final', '65a26da5419aaa32520219d9d1c5909769a7a19a88ff16e85df7d69c38028a55', 'pinned multilingual SHA')
need('c-stageA2-final', 'log_softmax', 'teacher-forced logprob scoring')
need('c-stageA2-final', 'eval determinism ok', 'repeat-logprob determinism gate')
need('c-stageA2-final', '1e-12', 'frozen-contrast reproduce tolerance')
need('c-stageA2-final', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
need('c-stageA2-final', 'CALIB_M2', 'calibrated IDX2 coefficient name')
need('c-stageA2-final', 'CALIB_M8', 'calibrated IDX8 coefficient name')
need('c-stageA2-final', '1.25', 'IDX2 1.25 level')
need('c-stageA2-final', '0.125', 'IDX8 0.125 level')
need('c-stageA2-final', '_set_gamma_pair(', 'per-site gamma scaling (in-memory)')
need('c-stageA2-final', '_restore_learned()', 'restore after every arm')
need('c-stageA2-final', 'calibrated-reader.json', 'canonical calibrated manifest')
need('c-stageA2-final', 'requires_grad_(False)', 'inference-only grad-free proof')
need('c-stageA2-final', 'no reader files written', 'no weight-saving discipline')
need('c-stageA2-final', '0.06166713312268257', '5M learned gamma IDX2 reference')
need('c-stageA2-final', '0.07537073642015457', '5M learned gamma IDX8 reference')
for name in ('hellaswag-disabled.jsonl', 'lambada-disabled.jsonl', 'hellaswag-raw.jsonl', 'lambada-raw.jsonl',
             'hellaswag-calibrated.jsonl', 'lambada-calibrated.jsonl', 'calibrated-reader.json',
             'config.json', 'checkpoint-shas.json', 'timings.json', 'summary.json', 'bootstrap.json'):
    need('c-stageA2-final', name, f'persist {name}')
tnb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
tcode = {c['id']: ''.join(c['source']) for c in tnb['cells'] if c['cell_type'] == 'code'}
for const in ['SOURCE_ID', 'TARGET_ID', 'MEM_DIM', 'HIDDEN', 'N_LAYERS', 'NGRAM', 'HEADS_PER_NGRAM',
              'ROW_DIM', 'ROWS_PER_PART', 'VOCAB_BASE', 'SEED', 'EOS', 'VOCAB', 'SEQ',
              'VAL_FAST_TOKENS', 'VAL_FULL_TOKENS', 'DATASET_ID', 'DATASET_CONFIG', 'BRANCHES', 'GAMMA_INIT']:
    assert const in code['c-config'] and const in tcode['c-config'], const
print('  ok [c-config] frozen constants present in both notebooks')
print('ALL A2-FINAL CHECKS PASSED')
