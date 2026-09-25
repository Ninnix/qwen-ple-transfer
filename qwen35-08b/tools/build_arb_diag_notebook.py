import json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
final = json.loads((REPO / 'kaggle_qwen35_08b_ple_eval_a2_final.ipynb').read_text(encoding='utf-8'))
stage_src = (REPO / 'tools' / 'arb_diag_stage_src.py').read_text(encoding='utf-8')
lines = stage_src.splitlines(keepends=True)
src_list = [l if l.endswith('\n') else l + '\n' for l in lines]
new_cells = []
for c in final['cells']:
    cid = c.get('id')
    if cid == 'c-stageA2-final':
        continue
    if cid == 'm-title':
        c = dict(c)
        c['source'] = ['# Qwen3.5-0.8B Stage A2-arb-diag: analysis-only alpha8 gate diagnostic (frozen B vs disabled baseline)\n']
        new_cells.append(c)
        continue
    new_cells.append(c)
new_cells.append({
    'id': 'c-stageA2-arb-diag',
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': src_list,
})
nb = dict(final)
nb['cells'] = new_cells
out = REPO / 'kaggle_qwen35_08b_ple_eval_a2_arb_diag.ipynb'
out.write_text(json.dumps(nb, indent=1) + '\n', encoding='utf-8')
print('wrote', out, 'cells:', len(new_cells))
