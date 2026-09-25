"""Static checks for the Stage A2-gamma diagnostic notebook: REAL-2M/3M localization
+ REAL-5M gamma-strength sweep + per-site ablation. Inference-only, frozen v34 reuse,
no training, no weight saving, no R4, logprob only."""
import ast
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_gamma.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA2-gamma' in code and 'c-evalh-a2' in code
def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')
def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')
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
print('  ok [all] no training token-count literals (only 2000384/3000320/5000192 actuals)')
need('c-stageA2-gamma', 'real-2m-r1.reader.safetensors', '2M weights file')
need('c-stageA2-gamma', 'real-3m-r1.reader.safetensors', '3M weights file')
need('c-stageA2-gamma', 'real-5m-r1.reader.safetensors', '5M weights file')
need('c-stageA2-gamma', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2-gamma', '_load_sfg(', 'safetensors loader with budget+SHA gate')
need('c-stageA2-gamma', '1ce2350cb7f00c82ee5acd4bf525ca1bfca743f9549a17386365ee63cea399f1', '2M byte-integrity SHA')
need('c-stageA2-gamma', '3aac4d72d757221a073444c42329192c25f9cc8a6d36752d551e193c8d7770ae', '3M byte-integrity SHA')
need('c-stageA2-gamma', '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594', '5M byte-integrity SHA')
need('c-stageA2-gamma', 'qwen-ple-a2-v34', 'frozen v34 results dataset')
need('c-stageA2-gamma', '2fc06364715b967f1860aea9cf38778875588b17', 'pinned tokenizer revision')
need('c-stageA2-gamma', '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'pinned hellaswag rev')
need('c-stageA2-gamma', '900124bf3b8235c6daf21033af9948b3f07346c4', 'pinned lambada rev')
need('c-stageA2-gamma', '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c', 'pinned full-val SHA')
need('c-stageA2-gamma', 'VAL_SUBSET_N', 'fixed val-NLL subset')
need('c-stageA2-gamma', 'log_softmax', 'teacher-forced logprob scoring')
need('c-stageA2-gamma', 'eval determinism ok', 'repeat-logprob determinism gate')
need('c-stageA2-gamma', '1e-12', 'frozen-contrast reproduce tolerance')
need('c-stageA2-gamma', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
need('c-stageA2-gamma', 'GAMMA_MULTS', 'gamma-strength sweep grid')
need('c-stageA2-gamma', '_set_gamma_both(', 'both-site gamma scaling (in-memory)')
need('c-stageA2-gamma', '_set_gamma_pair(', 'per-site gamma ablation (in-memory)')
need('c-stageA2-gamma', 'ABLATION', 'per-site ablation grid')
need('c-stageA2-gamma', 'requires_grad_(False)', 'inference-only grad-free proof')
need('c-stageA2-gamma', 'no reader files written', 'no weight-saving discipline')
need('c-stageA2-gamma', '0.06166713312268257', '5M learned gamma IDX2 reference')
need('c-stageA2-gamma', '0.07537073642015457', '5M learned gamma IDX8 reference')
for name in ('hellaswag-2m.jsonl', 'lambada-2m.jsonl', 'hellaswag-3m.jsonl', 'lambada-3m.jsonl',
             'sweep.json', 'ablation.json', 'config.json', 'checkpoint-shas.json', 'timings.json', 'summary.json', 'bootstrap.json'):
    need('c-stageA2-gamma', name, f'persist {name}')
tnb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
tcode = {c['id']: ''.join(c['source']) for c in tnb['cells'] if c['cell_type'] == 'code'}
for const in ['SOURCE_ID', 'TARGET_ID', 'MEM_DIM', 'HIDDEN', 'N_LAYERS', 'NGRAM', 'HEADS_PER_NGRAM',
              'ROW_DIM', 'ROWS_PER_PART', 'VOCAB_BASE', 'SEED', 'EOS', 'VOCAB', 'SEQ',
              'VAL_FAST_TOKENS', 'VAL_FULL_TOKENS', 'DATASET_ID', 'DATASET_CONFIG', 'BRANCHES', 'GAMMA_INIT']:
    assert const in code['c-config'] and const in tcode['c-config'], const
print('  ok [c-config] frozen constants present in both notebooks')
print('ALL A2-GAMMA DIAGNOSTIC CHECKS PASSED')
