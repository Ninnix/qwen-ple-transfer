"""Static checks for the Stage A2-arb-diag notebook: analysis-only alpha8 diagnostic.

Frozen REAL-5M R=1 (IDX 2+8), deployed-B gates vs disabled-baseline entropy.
No training of any kind: no optimizer, no backward, no weight writes, no R4.
Primary: alpha8 vs baseline entropy/confidence. Buckets: position, frequency,
punct/ws/newline, code keyword/ident, address reuse + trigram repeat."""
import ast
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_arb_diag.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA2-arb-diag' in code and 'c-evalh-a2' in code
assert 'c-stageA2-arb' not in code and 'c-stageA2-final' not in code and 'c-stageA2-calib' not in code and 'c-stageA2-gamma' not in code and 'c-stageA2-5m' not in code, 'stale stage cell leaked in'
print('  ok [stage] single arb-diag cell, no stale training/eval stage cell')
def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')
def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')
whole = '\n'.join(code.values())
diag = code['c-stageA2-arb-diag']
# scope: no training, no branch scaling, no heavy suites, no free generation
for banned in ('train_reader(', 'train_arb(', 'for R in [1, 4]', 'branches=4', 'run_downstream', 'MATH-500', 'LiveCodeBench', 'model.generate', '.generate(', 'real-500k-r1', 'real-1m-r1.reader.safetensors', 'resume.pt', 'save_file'):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no trainer, no R4, no heavy suites, no free generation')
for num in ('5000000', '2000000', '3000000', '1000000', '1000448', '500224'):
    assert num not in whole, f'token-count leak (training threshold): {num!r}'
print('  ok [all] no stale training token-count literals')
# analysis-only: optimizer/backward/weight-save must appear NOWHERE in the diag cell
for s in ('AdamW(', '.backward(', 'clip_grad_norm', 'torch.save', 'zero_grad', 'opt.step', 'loss.backward'):
    assert s not in diag, f'diag cell must be analysis-only, found {s!r}'
print('  ok [diag] analysis-only (no optimizer/backward/weight-save)')
# frozen pins (same as arb)
need('c-stageA2-arb-diag', 'real-5m-r1.reader.safetensors', '5M weights file')
need('c-stageA2-arb-diag', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2-arb-diag', '_load_sfc(', 'safetensors loader with budget+SHA gate')
need('c-stageA2-arb-diag', '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594', '5M byte-integrity SHA')
need('c-stageA2-arb-diag', 'qwen-ple-a2-v34', 'frozen v34 results dataset')
need('c-stageA2-arb-diag', 'qwen-ple-a2-arb', 'arb outputs dataset (B/C weights mount)')
need('c-stageA2-arb-diag', '595d195a84c65acd2aa414d61726e080b54801b4676b3d7a2256ca8a7bcd9948', 'arbitration-b.pt SHA pin')
need('c-stageA2-arb-diag', '2cbdd7913413c87ed45500d4496540fcbab30e3a97d6070a54e5f38d94eef3d9', 'arbitration-c.pt SHA pin')
need('c-stageA2-arb-diag', '2fc06364715b967f1860aea9cf38778875588b17', 'pinned tokenizer revision')
need('c-stageA2-arb-diag', '900124bf3b8235c6daf21033af9948b3f07346c4', 'pinned lambada rev')
need('c-stageA2-arb-diag', '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c', 'pinned full-val SHA')
need('c-stageA2-arb-diag', 'a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5', 'pinned general SHA')
need('c-stageA2-arb-diag', '735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421', 'pinned code SHA')
need('c-stageA2-arb-diag', 'c27f42f5f69de1aaf26020c7e14c2c141f8fa8490c5d95952c06e1c5bffa6648', 'pinned math SHA')
need('c-stageA2-arb-diag', 'c33632cf96170ca7ae92d4013702c14097fd79bb069c76e40fb3a531091936d8', 'pinned scientific SHA')
need('c-stageA2-arb-diag', '65a26da5419aaa32520219d9d1c5909769a7a19a88ff16e85df7d69c38028a55', 'pinned multilingual SHA')
# frozen discipline
need('c-stageA2-arb-diag', 'requires_grad_(False)', 'frozen grad-free proof')
need('c-stageA2-arb-diag', 'inference_mode', 'inference-only forwards')
need('c-stageA2-arb-diag', 'if self.memory is None:\n                return args, kw', 'disabled pass-through (pure-backbone baseline)')
need('c-stageA2-arb-diag', 'cap.set_memory(None)', 'stale-gate-memory clear before baseline')
need('c-stageA2-arb-diag', '0.06166713312268257', '5M learned gamma IDX2 reference')
need('c-stageA2-arb-diag', '0.07537073642015457', '5M learned gamma IDX8 reference')
need('c-stageA2-arb-diag', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
need('c-stageA2-arb-diag', 'eval determinism ok', 'repeat-forward determinism gate')
# gate math identical to arb deployment
need('c-stageA2-arb-diag', '0.5 * torch.sigmoid', 'alpha8_t 0.5*sigmoid form')
need('c-stageA2-arb-diag', 'h8.detach()', 'detached gate feature (no backbone graph for the gate)')
forbid('c-stageA2-arb-diag', 'rms_norm(h8.float())', 'attached gate feature (must detach h8)')
need('c-stageA2-arb-diag', 'B/C equivalence', 'B-only reporting justified by probe')
need('c-stageA2-arb-diag', 'no R4', 'explicit no-R4 discipline')
# primary analysis: entropy/confidence first
need('c-stageA2-arb-diag', 'alpha8-diag.json', 'diag manifest persistence')
need('c-stageA2-arb-diag', 'entropy', 'baseline entropy axis')
need('c-stageA2-arb-diag', 'spearman', 'rank correlation (monotone, outlier-robust)')
need('c-stageA2-arb-diag', 'pearson', 'linear correlation')
need('c-stageA2-arb-diag', 'maxprob', 'confidence axis')
need('c-stageA2-arb-diag', 'decile', 'decile bucketing')
need('c-stageA2-arb-diag', 'nll', 'true-token surprise axis')
# requested buckets
need('c-stageA2-arb-diag', 'position', 'position-in-sequence bucket')
need('c-stageA2-arb-diag', 'freq', 'token-frequency bucket')
need('c-stageA2-arb-diag', 'punct', 'punctuation class')
need('c-stageA2-arb-diag', 'whitespace', 'whitespace class')
need('c-stageA2-arb-diag', 'newline', 'newline class')
need('c-stageA2-arb-diag', 'keyword', 'code keyword class')
need('c-stageA2-arb-diag', 'ident', 'code identifier class')
need('c-stageA2-arb-diag', 'reuse', 'address-reuse flag')
need('c-stageA2-arb-diag', 'trigram', 'surface repetition flag')
need('c-stageA2-arb-diag', 'lambada', 'long-range lambada spans')
need('c-stageA2-arb-diag', 'hellaswag skipped', 'hellaswag exclusion reason')
for name in ('alpha8-diag.json', 'config-diag.json', 'timings-diag.json'):
    need('c-stageA2-arb-diag', name, f'persist {name}')
print('ALL A2-ARB-DIAG CHECKS PASSED')
