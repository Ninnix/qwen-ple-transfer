"""Static checks for the standalone Stage A2 eval notebook: eval-only (no training
paths), cheap MC logprob + domain NLL, mount-first checkpoints, controlled 500K
comparison + exploratory 1M, identical-settings harness, full persistence."""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA2' in code and 'c-evalh-a2' in code


def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')


def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')


whole = '\n'.join(code.values())

# No training paths anywhere. No heavy suites. No free generation (logprob only).
for banned in ('train_reader(', 'AdamW(', 'from torch.optim', '.backward(', 'clip_grad_norm', 'freeze_stream_tokens',
               'FineWebEduStream', 'SWEEP_ENABLED', 'BRANCH_ENABLED', 'REAL1M_ENABLED', 'REAL5M_ENABLED',
               'STAGEA_ENABLED', 'torch.save', 'for R in [1, 4]', "'optimizer':",
               't_math500', 't_lcb', 'run_downstream', 'MATH-500', 'LiveCodeBench',
               'model.generate', '.generate('):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no training paths, no R4/5M machinery, no heavy suites, no free generation')
assert '5000000' not in whole, '5M leak'
assert 'branches=4' not in whole, 'R4 leak'

# Sources: mount first, API fallback with clear-key auth.
need('c-stageA2', 'CKDIR = EFI_MOUNT', 'mount-first checkpoint source')
need('c-stageA2', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2', 'datasets', 'checkpoint download via Kaggle API')
need('c-stageA2', "assert h == run['sha256']", 'SHA gate on control checkpoints')
need('c-stageA2', 'need_sha', 'per-pair schema rules (bundles carry no stored sha)')
need('c-stageA2', '_load_sf(', 'safetensors path for kernel-written bundles (no stored sha)')
need('c-stageA2', 'def _solo(', 'single-active-arm hook discipline')
need('c-stageA2', 'log_softmax', 'teacher-forced logprob scoring')

# Controlled comparison, exploratory arm separate.
need('c-config', "ARMS_500K: tuple = ('disabled', 'random', 'permuted', 'real')", 'controlled 500K arms')
need('c-config', "ARM_1M: str = 'real-1m'", 'exploratory 1M arm')
need('c-stageA2', "('real', 'disabled'), ('real', 'random'), ('real', 'permuted'), ('real-1m', 'real')",
     'bootstrap contrasts (1M only vs 500K)')
forbid('c-stageA2', "'real-1m', 'disabled'", '1M mixed into causal comparison')

# Identical settings + determinism.
need('c-evalh-a2', 'EVAL_SEED = 1234', 'pinned eval seed')
need('c-stageA2', 'EVAL_BS', 'fixed batching')
need('c-stageA2', "tokenizer.padding_side = 'left'", 'fixed padding')
need('c-stageA2', 'eval determinism ok', 'repeat-logprob determinism gate')
need('c-evalh-a2', 'def _hellaswag_rows', 'HellaSwag adapter')
need('c-evalh-a2', 'def _piqa_rows', 'PIQA adapter')
need('c-evalh-a2', 'def _arc_rows', 'ARC-Easy adapter')
need('c-evalh-a2', 'def _lambada_rows', 'LAMBADA adapter')
need('c-evalh-a2', 'DOMAINS', '5-domain list')
need('c-evalh-a2', 'def build_domain_blocks', 'frozen domain builder')
need('c-evalh-a2', 'def load_domain', 'frozen domain loader')
need('c-evalh-a2', '_load_first(', 'candidate dataset resolution')

# Adaptive budget + hard guard + persistence.
need('c-stageA2', 'C.EVAL_TARGET_S', 'soft projection target')
need('c-stageA2', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
need('c-stageA2', 'projected total', 'early runtime projection')
for name in ('hellaswag.jsonl', 'piqa.jsonl', 'arc-easy.jsonl', 'lambada.jsonl',
             'config.json', 'checkpoint-shas.json', 'timings.json', 'summary.json', 'bootstrap.json'):
    need('c-stageA2', name, f'persist {name}')

# Frozen protocol constants identical to the train notebook.
tnb = json.loads((REPO / 'kaggle_qwen35_08b_ple_train.ipynb').read_text(encoding='utf-8'))
tcode = {c['id']: ''.join(c['source']) for c in tnb['cells'] if c['cell_type'] == 'code'}
for const in ['SOURCE_ID', 'TARGET_ID', 'MEM_DIM', 'HIDDEN', 'N_LAYERS', 'NGRAM', 'HEADS_PER_NGRAM',
              'ROW_DIM', 'ROWS_PER_PART', 'VOCAB_BASE', 'SEED', 'EOS', 'VOCAB', 'SEQ',
              'VAL_FAST_TOKENS', 'VAL_FULL_TOKENS', 'DATASET_ID', 'DATASET_CONFIG', 'BRANCHES', 'GAMMA_INIT']:
    assert const in code['c-config'] and const in tcode['c-config'], const
print('  ok [c-config] frozen constants present in both notebooks')
print('ALL A2 EVAL CHECKS PASSED')
