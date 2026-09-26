import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
sys.path.insert(0, str(TRACK.parent / 'qwen35-08b/lightning'))
import bootstrap


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def contrast(a, b, name):
    assert len(a['val_blocks']) == len(b['val_blocks']) == 1024
    assert len(a['hs_correct']) == len(b['hs_correct']) == 1000
    assert len(a['lam_correct']) == len(b['lam_correct']) == 1000
    assert len(a['lam_nlls']) == len(b['lam_nlls']) == 1000
    for x, y in zip(a['val_blocks'], b['val_blocks']):
        assert x['n'] == y['n'] == 511
    out = bootstrap.contrast(name, a, b)
    dd = {}
    for d in sorted(a['dom_blocks']):
        x, y = a['dom_blocks'][d], b['dom_blocks'][d]
        assert len(x) == len(y) == 64
        assert all(i['n'] == j['n'] == 511 for i, j in zip(x, y))
        dd[d] = [i['sum']/i['n'] - j['sum']/j['n'] for i, j in zip(x, y)]
    rng = random.Random(1234)
    reps = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v)/len(v)
                      for v in dd.values())/len(dd) for _ in range(10000))
    out['domain_mean_nll'] = {
        'mean': sum(sum(v)/len(v) for v in dd.values())/len(dd),
        'lo': reps[250], 'hi': reps[9749],
        'p': min(1.0, 2 * min(sum(v >= 0 for v in reps), sum(v <= 0 for v in reps))/10000),
        'n': 320,
    }
    out['paired_deltas'] = {
        'val_nll': [x['sum']/x['n'] - y['sum']/y['n'] for x, y in zip(a['val_blocks'], b['val_blocks'])],
        'domains': dd,
        'hs_acc': [x-y for x, y in zip(a['hs_correct'], b['hs_correct'])],
        'lambada_acc': [x-y for x, y in zip(a['lam_correct'], b['lam_correct'])],
        'lambada_nll': [x-y for x, y in zip(a['lam_nlls'], b['lam_nlls'])],
    }
    return out


def fields(e):
    return {'full_val_nll': e['val_nll'], **e['domains'],
            'domain_mean_nll': e['domain_mean'], 'lambada_nll': e['lambada_nll'],
            'lambada_acc': e['lambada_acc'], 'hs_acc': e['hs_acc']}


def ci_rows(c):
    return {'full_val_nll': c['val_nll'], **c['domains'],
            **{k: c[k] for k in ('domain_mean_nll', 'lambada_nll', 'lambada_acc', 'hs_acc')}}


def main():
    paths = {
        'raw10': HERE / 'results/eval-raw-10m.json',
        'linear10': HERE / 'results/eval-linear750.json',
        'raw15': TRACK / 'results/arb-eval-raw-15m.json',
        'linear15': TRACK / 'results/arb-eval-linear750.json',
    }
    old_manifest = json.loads((TRACK / 'results/milestone-shas.json').read_text())
    for key in ('raw15', 'linear15'):
        assert sha(paths[key]) == old_manifest[paths[key].name]['sha256']
    stock_diff = json.loads((HERE / 'results/stock-reproduction.json').read_text())
    assert max(stock_diff.values()) <= 2e-6
    evals = {k: json.loads(p.read_text()) for k, p in paths.items()}
    summaries = {k: fields(e) for k, e in evals.items()}
    contrasts = {mode: contrast(evals[mode+'15'], evals[mode+'10'], mode+' 15M - 10M')
                 for mode in ('raw', 'linear')}
    calibrated = contrasts['linear']
    reasons = []
    for k in ('val_nll', 'domain_mean_nll'):
        if calibrated[k]['hi'] >= 0:
            reasons.append(k + ': 15M improvement is not clear at 95%')
    for name, metric in {**calibrated['domains'], 'lambada_nll': calibrated['lambada_nll']}.items():
        if metric['lo'] > 0:
            reasons.append(name + ': clear NLL regression at 15M')
    for k in ('hs_acc', 'lambada_acc'):
        if calibrated[k]['mean'] < 0:
            reasons.append(k + ': 15M accuracy point estimate is lower')
    canonical = 10 if reasons else 15
    result = {
        'protocol': json.loads((HERE / 'protocol.json').read_text()),
        'protocol_sha256': sha(HERE / 'protocol.json'),
        'input_sha256': {k: sha(p) for k, p in paths.items()},
        'evaluation_streams_sha256': sha(TRACK / 'results/evaluation-streams.json'),
        'stock_reproduction_max_diff': stock_diff,
        'metrics': summaries, 'contrasts_15m_minus_10m': contrasts,
        'canonical_balanced_milestone_m': canonical, 'selection_reasons': reasons,
    }
    (HERE / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Qwengram-2B: matched 10M versus 15M', '',
             '**Canonical balanced endpoint: REAL-%dM + linear750.**' % canonical, '',
             'Both readers are persisted milestones. Only the missing 10M arbiter was trained, '
             'using exactly the existing 749,568-token calibration protocol. The 15M arbiter '
             'and its SHA-verified evaluations were reused. No reader training was performed.', '',
             '| Metric | Raw 10M | Calibrated 10M | Raw 15M | Calibrated 15M |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in summaries['raw10']:
        acc = name.endswith('_acc')
        vals = [summaries[k][name] for k in ('raw10', 'linear10', 'raw15', 'linear15')]
        cells = ['%.2f%%' % (100*v) if acc else '%.6f' % v for v in vals]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired 15M minus 10M', '',
              'Negative NLL favors 15M; positive accuracy favors 15M. Accuracy deltas '
              'are percentage points. Brackets give paired 95% confidence intervals.', '',
              '| Metric | Raw delta [95% CI] | Calibrated delta [95% CI] |',
              '| --- | ---: | ---: |']
    rows = {k: ci_rows(v) for k, v in contrasts.items()}
    for name in rows['raw']:
        factor = 100 if name.endswith('_acc') else 1
        cells = []
        for mode in ('raw', 'linear'):
            m = rows[mode][name]
            cells.append('%+.6f [%+.6f, %+.6f]' % tuple(factor*m[k] for k in ('mean', 'lo', 'hi')))
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    stock = json.loads((TRACK / 'results/arb-eval-stock.json').read_text())
    lines += ['', '## Endpoint decision', '']
    if canonical == 10:
        lines += ['15M does not meet the predeclared balanced-improvement rule. Retain '
                  '**10M + linear750** as canonical and **15M as the max-LM/research endpoint**.', '']
        lines += ['15M clearly improves full-val, domain-mean and LAMBADA NLL, and four '
                  'individual domains; code remains inconclusive. Neither benchmark accuracy '
                  'difference is resolved. The calibrated HellaSwag decrease is **not a '
                  'statistically established regression**. Choosing 10M follows the conservative '
                  'selection rule, not evidence that 10M is statistically superior overall.', '']
    else:
        lines += ['15M meets the predeclared balanced-improvement rule; retain '
                  '**15M + linear750** as canonical.', '']
    lines += ['- ' + r for r in reasons]
    lines += ['']
    for key in ('linear10', 'linear15'):
        nll = evals[key]['val_nll']
        lines.append('%s: full-val perplexity %.6f, %.3f%% below stock (%.6f).' %
                     (key, math.exp(nll), 100*(1-math.exp(nll-stock['val_nll'])), math.exp(stock['val_nll'])))
        lines.append('')
    lines += ['## Method and provenance', '',
              'The same frozen streams and example order are used for every comparison: '
              '1,024 full-validation blocks, 64 blocks in each of five domains, '
              '1,000 HellaSwag examples and 1,000 LAMBADA examples. The original bootstrap '
              'uses 10,000 paired resamples with seed 1234. Domain mean is an equal-domain '
              'stratified bootstrap. Intervals are marginal, without multiplicity adjustment. '
              'Overlapping intervals do not establish equivalence, and the selected checkpoint '
              'still needs matched runtime validation.', '',
              'The repeated stock evaluation passed the predeclared 2e-6 per-block/per-example '
              'NLL tolerance and identical accuracy-array checks.', '',
              'See [protocol](protocol.json), [complete paired data](comparison.json), '
              'and [10M evaluation outputs](results/).', '']
    (HERE / 'decision.md').write_text('\n'.join(lines))
    print(json.dumps({'canonical': canonical, 'reasons': reasons, 'metrics': summaries}, indent=2))


if __name__ == '__main__':
    main()
