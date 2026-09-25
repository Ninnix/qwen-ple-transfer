"""Static checks for the standalone Stage A eval notebook: eval-only (no training
paths), mount-first checkpoints with API fallback, controlled 500K comparison +
exploratory 1M, identical-settings harness, full persistence."""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA' in code and 'c-evalh' in code


def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')


def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')


whole = '\n'.join(code.values())

# No training paths anywhere.
for banned in ('train_reader(', 'AdamW(', 'from torch.optim', '.backward(', 'clip_grad_norm', 'freeze_stream_tokens',
               'FineWebEduStream', 'SWEEP_ENABLED', 'BRANCH_ENABLED', 'REAL1M_ENABLED',
               'STAGEA_ENABLED', 'torch.save', 'for R in [1, 4]', "'optimizer':",
               't_math500', 't_hellaswag', 't_lcb', 'run_downstream', 'MATH-500', 'LiveCodeBench'):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no training paths, no R4/5M machinery, no MATH/LCB')
assert '5000000' not in whole, '5M leak'

# Sources: mount first, API fallback with clear-key auth.
need('c-stageA', 'CKDIR = EFI_MOUNT', 'mount-first checkpoint source')
need('c-stageA', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA', 'datasets', 'checkpoint download via Kaggle API')
need('c-stageA', "assert h == run['sha256']", 'SHA gate on control checkpoints')
need('c-stageA', 'need_sha', 'per-pair schema rules (bundles carry no stored sha)')
need('c-stageA', '_load_sf(', 'safetensors path for kernel-written bundles (no stored sha)')
need('c-stageA', 'def _solo(', 'single-active-arm hook discipline')
need('c-stageA', 'past_key_values', 'KV-cache stepwise decode with per-step memory')
need('c-stageA', 'arm/metric mismatch', 'per-block end-to-end arm validity')

# Controlled comparison, exploratory arm separate.
need('c-config', "ARMS_500K: tuple = ('disabled', 'random', 'permuted', 'real')", 'controlled 500K arms')
need('c-config', "ARM_1M: str = 'real-1m'", 'exploratory 1M arm')
need('c-stageA', "('real', 'disabled'), ('real', 'random'), ('real', 'permuted'), ('real-1m', 'real')",
     'bootstrap contrasts (1M only vs 500K)')
forbid('c-stageA', "'real-1m', 'disabled'", '1M mixed into causal comparison')

# Identical settings + determinism.
need('c-evalh', 'do_sample=False', 'greedy everywhere')
need('c-evalh', "tokenizer.padding_side = 'left'", 'fixed padding')
need('c-evalh', 'EVAL_BS', 'fixed batching')
need('c-evalh', 'EVAL_SEED = 1234', 'pinned eval seed')
need('c-stageA', 'eval determinism ok', 'repeat-generation determinism gate')
need('c-evalh', 'def t_mmlupro(arms, budget_ts, n=2000):', 'MMLU-Pro subset param')
need('c-evalh', 'def t_simpleqa(arms, budget_ts, n=1000):', 'SimpleQA subset param')
need('c-evalh', '_load_first(', 'candidate dataset resolution')
need('c-evalh', 'def t_humaneval(arms, budget_ts):', 'HumanEval+ complete')
need('c-evalh', 'def dump_perblock(', 'per-block NLL dumper')

# Adaptive budget + hard guard + persistence.
need('c-stageA', 'C.EVAL_TARGET_S', 'soft projection target')
need('c-stageA', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
need('c-stageA', 'projected total', 'early runtime projection')
for name in ('predictions.jsonl', 'perblock-nll.jsonl', 'config.json', 'checkpoint-shas.json',
             'validation-sha.json', 'timings.json', 'summary.json', 'bootstrap.json'):
    need('c-stageA', name, f'persist {name}')

# Frozen protocol constants identical to the train notebook.
tnb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
tcode = {c['id']: ''.join(c['source']) for c in tnb['cells'] if c['cell_type'] == 'code'}
for const in ['SOURCE_ID', 'TARGET_ID', 'MEM_DIM', 'HIDDEN', 'N_LAYERS', 'NGRAM', 'HEADS_PER_NGRAM',
              'ROW_DIM', 'ROWS_PER_PART', 'VOCAB_BASE', 'SEED', 'EOS', 'VOCAB', 'SEQ',
              'VAL_FAST_TOKENS', 'VAL_FULL_TOKENS', 'DATASET_ID', 'DATASET_CONFIG', 'BRANCHES', 'GAMMA_INIT']:
    assert const in code['c-config'] and const in tcode['c-config'], const
print('  ok [c-config] frozen constants present in both notebooks')

# Kernel metadata may select another frozen eval notebook.
km = json.loads((REPO / 'kernel-metadata.json').read_text(encoding='utf-8'))
assert km['code_file'].startswith('kaggle_qwen35_08b_ple_eval'), km['code_file']
assert (REPO / km['code_file']).exists(), km['code_file']
assert km['id'] == 'ninnix/qwen3-5-0-8b-ple-target-side-reader-p100', km['id']
assert km['machine_shape'] == 'NvidiaTeslaP100' and km['enable_gpu'] == 'true'
print('  ok [metadata] kernel selects a present eval notebook')
print('ALL EVAL CHECKS PASSED')
