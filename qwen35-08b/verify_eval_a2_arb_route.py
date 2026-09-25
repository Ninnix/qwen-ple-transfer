"""Static checks for the Stage A2-arb-route notebook: inference-only causal routing
diagnostic on frozen REAL-5M R=1 (IDX 2+8) with deployed-B gates.

Arms: learned / shuffled / hard (top-a8 to top-entropy) / easy (top-a8 to
bottom-entropy). Per-sequence alpha8 multiset exact in every arm; IDX2 fixed
1.25. Eval: full-val NLL, hellaswag acc, lambada acc/NLL, 5 frozen domains.
Learned arm must reproduce arb-B (invariant). No training of any kind: no
optimizer, no backward, no weight writes, no R4."""
import ast
import json
from pathlib import Path
REPO = Path(__file__).resolve().parent
nb = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_arb_route.ipynb').read_text(encoding='utf-8'))
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
assert 'c-stageA2-arb-route' in code and 'c-evalh-a2' in code
assert 'c-stageA2-arb' not in code and 'c-stageA2-arb-diag' not in code and 'c-stageA2-final' not in code and 'c-stageA2-calib' not in code and 'c-stageA2-gamma' not in code and 'c-stageA2-5m' not in code, 'stale stage cell leaked in'
print('  ok [stage] single arb-route cell, no stale stage cell')
def need(cell, sub, msg):
    assert sub in code[cell], f'{cell}: missing {msg}'
    print(f'  ok [{cell}] {msg}')
def forbid(cell, sub, msg):
    assert sub not in code[cell], f'{cell}: forbidden {msg}'
    print(f'  ok [{cell}] no {msg}')
whole = '\n'.join(code.values())
diag = code['c-stageA2-arb-route']
# scope: no training, no branch scaling, no heavy suites, no free generation
for banned in ('train_reader(', 'train_arb(', 'for R in [1, 4]', 'branches=4', 'run_downstream', 'MATH-500', 'LiveCodeBench', 'model.generate', '.generate(', 'real-500k-r1', 'real-1m-r1.reader.safetensors', 'resume.pt', 'save_file'):
    assert banned not in whole, f'training/scope leak: {banned!r}'
print('  ok [all] no trainer, no R4, no heavy suites, no free generation')
for num in ('5000000', '2000000', '3000000', '1000000', '1000448', '500224'):
    assert num not in whole, f'token-count leak (training threshold): {num!r}'
print('  ok [all] no stale training token-count literals')
# analysis-only: optimizer/backward/weight-save must appear NOWHERE in the route cell
for s in ('AdamW(', '.backward(', 'clip_grad_norm', 'torch.save', 'zero_grad', 'opt.step', 'loss.backward'):
    assert s not in diag, f'route cell must be analysis-only, found {s!r}'
print('  ok [route] analysis-only (no optimizer/backward/weight-save)')
# frozen pins (same as arb)
need('c-stageA2-arb-route', 'real-5m-r1.reader.safetensors', '5M weights file')
need('c-stageA2-arb-route', 'KAGGLE_API_TOKEN', 'API fallback auth (clear key)')
need('c-stageA2-arb-route', '_load_sfc(', 'safetensors loader with budget+SHA gate')
need('c-stageA2-arb-route', '078d7b47b979dbdae1442cbcc15bc466f72b8942159ede8c02fad0daedb63594', '5M byte-integrity SHA')
need('c-stageA2-arb-route', 'qwen-ple-a2-v34', 'frozen v34 results dataset')
need('c-stageA2-arb-route', 'qwen-ple-a2-arb', 'arb outputs dataset (B weights mount)')
need('c-stageA2-arb-route', '595d195a84c65acd2aa414d61726e080b54801b4676b3d7a2256ca8a7bcd9948', 'arbitration-b.pt SHA pin')
need('c-stageA2-arb-route', '2fc06364715b967f1860aea9cf38778875588b17', 'pinned tokenizer revision')
need('c-stageA2-arb-route', '218ec52e09a7e7462a5400043bb9a69a41d06b76', 'pinned hellaswag rev')
need('c-stageA2-arb-route', '900124bf3b8235c6daf21033af9948b3f07346c4', 'pinned lambada rev')
need('c-stageA2-arb-route', '26ffe65b1f6457d2557dbacc98197d025932057e3c183e3cf1de8eefb7c2fe7c', 'pinned full-val SHA')
need('c-stageA2-arb-route', 'a9e12294eba002b5a4e9fce610ff39f8aed6d802e36e534e82ed367f96bb9aa5', 'pinned general SHA')
need('c-stageA2-arb-route', '735f933ca4bc694562dbb2a6ab9c227c802268f7715448eba50e172ae450c421', 'pinned code SHA')
need('c-stageA2-arb-route', 'c27f42f5f69de1aaf26020c7e14c2c141f8fa8490c5d95952c06e1c5bffa6648', 'pinned math SHA')
need('c-stageA2-arb-route', 'c33632cf96170ca7ae92d4013702c14097fd79bb069c76e40fb3a531091936d8', 'pinned scientific SHA')
need('c-stageA2-arb-route', '65a26da5419aaa32520219d9d1c5909769a7a19a88ff16e85df7d69c38028a55', 'pinned multilingual SHA')
# frozen discipline + redirect fixes
need('c-stageA2-arb-route', 'requires_grad_(False)', 'frozen grad-free proof')
need('c-stageA2-arb-route', 'inference_mode', 'inference-only forwards')
need('c-stageA2-arb-route', "mode = 'off'", 'disabled mode (pure-backbone baseline)')
need('c-stageA2-arb-route', 'rt.memory = None', 'stale-gate-memory clear before baseline')
need('c-stageA2-arb-route', 'to(lg.device)', 'device-matched gather index (cuda logits)')
need('c-stageA2-arb-route', '0.06166713312268257', '5M learned gamma IDX2 reference')
need('c-stageA2-arb-route', '0.07537073642015457', '5M learned gamma IDX8 reference')
need('c-stageA2-arb-route', 'C.EVAL_HARD_S - 1800', 'hard wall guard with persist margin')
need('c-stageA2-arb-route', 'eval determinism ok', 'repeat-forward determinism gate')
# gate math identical to arb deployment
need('c-stageA2-arb-route', '0.5 * torch.sigmoid', 'alpha8_t 0.5*sigmoid form')
need('c-stageA2-arb-route', 'h8.detach()', 'detached gate feature (no backbone graph for the gate)')
forbid('c-stageA2-arb-route', 'rms_norm(h8.float())', 'attached gate feature (must detach h8)')
need('c-stageA2-arb-route', 'no R4', 'explicit no-R4 discipline')
# routing arms: budget exact, placement only
need('c-stageA2-arb-route', "'learned', 'shuffled', 'hard', 'easy'", 'four-arm tuple')
need('c-stageA2-arb-route', 'randperm', 'pinned per-sequence shuffle')
need('c-stageA2-arb-route', 'manual_seed(1234)', 'pinned shuffle seed')
need('c-stageA2-arb-route', 'multiset exact', 'budget discipline label')
need('c-stageA2-arb-route', 'per-arm alpha8 budget diverged', 'budget-match assert')
need('c-stageA2-arb-route', 'entropy', 'disabled-entropy ranking axis')
# eval coverage + learned==B invariant
need('c-stageA2-arb-route', 'val_full_chk', 'full-val NLL coverage')
need('c-stageA2-arb-route', 'hellaswag', 'hellaswag coverage')
need('c-stageA2-arb-route', 'lambada', 'lambada coverage')
need('c-stageA2-arb-route', 'learned arm does not reproduce arb-B', 'learned==B invariant')
need('c-stageA2-arb-route', '2.865490686403562', 'arb-B val NLL reference')
need('c-stageA2-arb-route', '2.253693464072421', 'arb-B lambada NLL reference')
need('c-stageA2-arb-route', 'routing.json', 'routing manifest persistence')
need('c-stageA2-arb-route', 'bootstrap.json', 'paired-contrast persistence')
for name in ('routing.json', 'config-route.json', 'timings-route.json'):
    need('c-stageA2-arb-route', name, f'persist {name}')
print('ALL A2-ARB-ROUTE CHECKS PASSED')
