"""Static checks for the incremental Stage A2-5M eval notebook: REAL-5M only,
frozen v34 reuse (manifests, SHAs, revs, union), merged persistence, no training,
no other-arm reruns, no R4, logprob only."""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_5m.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA2-5m' in code and 'c-evalh-a2' in code


def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')


def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')


whole = '\n'.join(code.values())

# No training paths, no other-arm checkpoint reruns, no R4, logprob only.
for banned in ('train_reader(', 'AdamW(', 'from torch.optim', '.backward(', 'clip_grad_norm', 'freeze_stream_tokens',
               'FineWebEduStream', 'SWEEP_ENABLED', 'BRANCH_ENABLED', 'REAL1M_ENABLED', 'REAL5M_ENABLED',
               'STAGEA_ENABLED', 'torch.save', 'for R in [1, 4]', "'optimizer':",
               'run_downstream', 'MATH-500', 'LiveCodeBench',
               'model.generate', '.generate(',
               'reader-control-random', 'reader-control-permuted', 'real-500k-r1', 'real-1m-r1.reader.safetensors',
               'resume.pt', 'branches=4'):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no training, no other-arm reruns, no R4, no heavy suites, no free generation')
assert '5000000' not in whole, '5M token-count leak (only 5000192 expected)'

# REAL-5M checkpoint source + integrity.
need('c-stageA2-5m', 'real-5m-r1.reader.safetensors', '5M weights file')
need('c-stageA2-5m', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2-5m', '_load_sf5(', 'safetensors loader with budget gate')
need('c-stageA2-5m', 'safetensors_sha256', 'byte-integrity gate on 5M weights')
need('c-stageA2-5m', 'qwen-ple-a2-v34', 'frozen v34 results dataset')

# Frozen reuse gates.
need('c-stageA2-5m', '2fc06364715b967f1860aea9cf38778875588b17', 'pinned tokenizer revision')
need('c-stageA2-5m', '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'pinned hellaswag rev')
need('c-stageA2-5m', '210d026faf9955653af8916fad021475a3f00453', 'pinned arc rev')
need('c-stageA2-5m', '900124bf3b8235c6daf21033af9948b3f07346c4', 'pinned lambada rev')
need('c-stageA2-5m', 'a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5', 'pinned general SHA')
need('c-stageA2-5m', '735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421', 'pinned code SHA')
need('c-stageA2-5m', '3934003', 'pinned union address count')
need('c-stageA2-5m', 'log_softmax', 'teacher-forced logprob scoring')
need('c-stageA2-5m', 'eval determinism ok', 'repeat-logprob determinism gate')
need('c-stageA2-5m', "('real-5m', 'real-1m'), ('real-5m', 'disabled')", 'new 5M bootstrap contrasts')
need('c-stageA2-5m', '1e-12', 'frozen-contrast reproduce tolerance')
need('c-stageA2-5m', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
for name in ('hellaswag.jsonl', 'arc-easy.jsonl', 'lambada.jsonl',
             'config.json', 'checkpoint-shas.json', 'timings.json', 'summary.json', 'bootstrap.json'):
    need('c-stageA2-5m', name, f'persist {name}')

# Frozen protocol constants identical to the train notebook.
tnb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
tcode = {c['id']: ''.join(c['source']) for c in tnb['cells'] if c['cell_type'] == 'code'}
for const in ['SOURCE_ID', 'TARGET_ID', 'MEM_DIM', 'HIDDEN', 'N_LAYERS', 'NGRAM', 'HEADS_PER_NGRAM',
              'ROW_DIM', 'ROWS_PER_PART', 'VOCAB_BASE', 'SEED', 'EOS', 'VOCAB', 'SEQ',
              'VAL_FAST_TOKENS', 'VAL_FULL_TOKENS', 'DATASET_ID', 'DATASET_CONFIG', 'BRANCHES', 'GAMMA_INIT']:
    assert const in code['c-config'] and const in tcode['c-config'], const
print('  ok [c-config] frozen constants present in both notebooks')
print('ALL A2-5M EVAL CHECKS PASSED')
