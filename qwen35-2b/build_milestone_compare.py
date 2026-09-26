import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'milestone-compare'
OUT.mkdir(exist_ok=True)
reference = ROOT / 'arb-eval/arb_eval.ipynb'
nb = json.loads(reference.read_text())
original = {c['id']: ''.join(c['source']) for c in nb['cells']}
for c in nb['cells']:
    body = original[c['id']]
    if c['id'] == 'c-install':
        body = body.replace('datasets safetensors huggingface_hub',
                            'datasets==5.0.0 safetensors==0.8.0 huggingface_hub==1.11.0')
        body += "\nassert torch.__version__ == '2.10.0+cu128'\n"
    if c['id'] == 'c-reader-load':
        body = body.replace('15m', '10m').replace('15000064', '10000384').replace('15M', '10M')
        body = body.replace('reader-train-10m.json', 'reader-train-15m.json')
        body += "\nassert sha(reader_path) == '4f93cd735bd7cbe99d3d35e364b70222ee56b6c27574f12822ee58e143cd8feb'\n"
        body += "assert sha(resume_path) == '63d23c3bf66c41c97f16da6a68d42bd6d7e9f3302e61e3a42b0ec88a5e905b55'\n"
    if c['id'] == 'c-evaluation':
        body = body.replace('raw-15m', 'raw-10m')
        check = '''
prior_dir = summary_hits[0].parent
prior_manifest = json.loads((prior_dir / 'milestone-shas.json').read_text())
for filename in ('arb-eval-stock.json', 'arb-eval-raw-15m.json', 'arb-eval-linear750.json'):
    assert sha(prior_dir / filename) == prior_manifest[filename]['sha256']
old_stock = json.loads((prior_dir / 'arb-eval-stock.json').read_text())
assert stock['hs_correct'] == old_stock['hs_correct']
assert stock['lam_correct'] == old_stock['lam_correct']
def block_diff(a, b):
    assert len(a) == len(b)
    assert all(x['n'] == y['n'] for x, y in zip(a, b))
    return max(abs(x['sum']/x['n'] - y['sum']/y['n']) for x, y in zip(a, b))
stock_diff = {'full': block_diff(stock['val_blocks'], old_stock['val_blocks']),
              **{d: block_diff(stock['dom_blocks'][d], old_stock['dom_blocks'][d]) for d in domains},
              'lambada': max(abs(x-y) for x, y in zip(stock['lam_nlls'], old_stock['lam_nlls']))}
assert max(stock_diff.values()) <= 2e-6, stock_diff
(work / 'stock-reproduction.json').write_text(json.dumps(stock_diff, indent=2))
print('STOCK_REPRODUCED', stock_diff, flush=True)
'''
        body = body.replace("stock = score_static('stock', True)\n", "stock = score_static('stock', True)\n" + check)
    clean = '\n'.join(s for s in body.splitlines() if not s.startswith('%'))
    ast.parse(clean)
    c['source'] = body.splitlines(keepends=True)
    c['execution_count'] = None
    c['outputs'] = []

after = {c['id']: ''.join(c['source']) for c in nb['cells']}
for name in ('c-config', 'c-hashing', 'c-reader', 'c-inject', 'c-ple', 'c-tok',
             'c-load', 'c-data', 'c-valprep', 'c-cache-arb', 'c-arb-math', 'c-train-arb'):
    assert original[name] == after[name], name
(OUT / 'compare.ipynb').write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / 'arb-eval/kernel-metadata.json').read_text())
meta.update(code_file='compare.ipynb', id='ninnix/qwengram-2b-10m-linear750-comparison',
            title='Qwengram 2B 10M linear750 comparison')
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
protocol = {
    'reference_notebook_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
    'comparison_notebook_sha256': hashlib.sha256((OUT / 'compare.ipynb').read_bytes()).hexdigest(),
    'reader_tokens': [10000384, 15000064],
    'calibration_tokens': 749568,
    'new_training': 'Only missing 10M linear arbiter; reader and backbone frozen',
    'evaluation': 'Same frozen full validation, five domains, HellaSwag-1000 and LAMBADA-1000',
    'existing_15m_results': 'SHA-verified persisted per-block/per-example outputs; no retraining',
    'stock_reproduction': 'Identical accuracy arrays; maximum per-block/per-example NLL difference <= 2e-6',
    'contrasts': ['raw_15m_minus_raw_10m', 'linear_15m_minus_linear_10m'],
    'bootstrap': {'replicates': 10000, 'seed': 1234, 'confidence': 0.95,
                  'method': 'Original paired per-block/per-example bootstrap; domain mean stratified equally across five domains',
                  'intervals': 'Marginal percentile intervals, no multiplicity adjustment'},
    'balanced_15m_rule': {
        'full_val_and_domain_mean': 'Both calibrated 15M-minus-10M NLL interval upper bounds < 0',
        'other_nll_metrics': 'No five-domain or LAMBADA NLL interval wholly above zero',
        'accuracy': 'HellaSwag and LAMBADA point estimates must not fall; no interval wholly below zero',
        'otherwise': '10M canonical balanced endpoint; 15M max-LM/research endpoint',
        'scope': 'Conservative operational selection rule; overlapping intervals do not prove equivalence'
    }
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
print('10M comparison notebook built; arbitration math and training code unchanged')
