import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
R = ROOT / 'results'


def read(name):
    return json.loads((R / name).read_text())


def ci(x):
    return '%+.6f [%+.6f, %+.6f]' % (x.get('mean', x.get('delta')), x['lo'], x['hi'])


s = read('arb-summary.json')
b = read('arb-bootstrap.json')
c = read('control-controls-500k-summary.json')
m = read('milestone-shas.json')
runtime = read('runtime.json')
domains = ('general', 'code', 'math', 'scientific', 'multilingual')
arms = ('stock', 'raw', 'linear750')
lines = ['# Qwengram-2B measured results', '',
         'All NLLs use natural logarithms. Lower NLL is better. Accuracy is a fraction.', '',
         '## Main evaluation', '',
         '| Arm | Full-val NLL | Delta vs stock | Perplexity | PPL reduction | Five-domain mean | HellaSwag-1000 | LAMBADA-1000 | LAMBADA NLL |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for arm in arms:
    x = s[arm]
    delta = x['val_nll'] - s['stock']['val_nll']
    lines.append('| %s | %.6f | %+.6f | %.6f | %.3f%% | %.6f | %.3f | %.3f | %.6f |' %
                 (arm, x['val_nll'], delta, math.exp(x['val_nll']), 100 * (1 - math.exp(delta)),
                  x['domain_mean'], x['hs_acc'], x['lambada_acc'], x['lambada_nll']))
lines += ['', '| Domain | Stock | Raw 15M | Linear750 | Linear750 − stock [95% CI] |',
          '| --- | ---: | ---: | ---: | --- |']
for d in domains:
    lines.append('| %s | %.6f | %.6f | %.6f | %s |' %
                 (d, *(s[a]['domains'][d] for a in arms), ci(b['linear_minus_stock']['domains'][d])))
lines += ['', '## Paired bootstrap', '',
          '10,000 paired resamples, seed 1234: 1,024 full-val blocks, 64 blocks per domain,',
          'and 1,000 aligned examples per benchmark. Domain-mean resampling preserves the',
          'five-domain balance. The bootstrap code and scoring definitions are the frozen',
          '0.8B implementation. All per-item deltas are in `arb-bootstrap.json`.', '',
          '| Metric | Raw − stock | Linear750 − stock | Linear750 − raw |',
          '| --- | --- | --- | --- |']
for key in ('val_nll', 'domain_mean_nll', 'hs_acc', 'lambada_acc', 'lambada_nll'):
    lines.append('| %s | %s | %s | %s |' %
                 (key, *(ci(b[x][key]) for x in ('raw_minus_stock', 'linear_minus_stock', 'linear_minus_raw'))))
lines += ['', '## Arbitration and reader diagnostics', '',
          '| Arm | Early alpha | Late mean | Late std | Late p05 | Late p50 | Late p95 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
          '| stock | disabled | — | — | — | — | — |',
          '| raw | 1 | 1 | 0 | 1 | 1 | 1 |']
a = s['linear750']['alpha8']['full-val']
lines.append('| linear750 | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |' %
             (s['early_alpha'], *(a[k] for k in ('mean', 'std', 'p05', 'p50', 'p95'))))
lines += ['', 'Late alpha is token dependent; the table uses all full-val tokens. Statistics',
          'for every domain and benchmark are retained in `arb-summary.json`.', '',
          '| Site | Trained gamma (raw and final) | Raw effective strength | Final effective strength |',
          '| --- | ---: | ---: | ---: |',
          '| IDX2 | %.6f | %.6f | %.6f |' % (s['gammas']['2'], s['gammas']['2'], s['effective_early']),
          '| IDX8 | %.6f | %.6f | %.6f (mean) |' % (s['gammas']['8'], s['gammas']['8'], s['effective_late_mean']),
          '', 'Stock effective strengths are zero. Reader parameter updates from the common',
          'initialization are identical for raw and final because arbitration freezes the reader:',
          '', '| Tensor | L2 update norm |', '| --- | ---: |']
for name, value in s['reader_parameter_update_norms'].items():
    lines.append('| %s | %.6f |' % (name, value))
lines += ['', '| Site | Reader residual L2 | After arbitration L2 | Relative to hidden L2 |',
          '| --- | ---: | ---: | ---: |']
for site, vals in s['reader_residual_update_norms'].items():
    lines.append('| %s | %.6f | %.6f | %.6f |' %
                 (site, vals['reader'], vals['effective'], vals['relative']))
lines += ['', 'Residual diagnostics average the first 64 full-val blocks at the final gated',
          'model’s hidden states. They are distinct from the parameter-update norms.', '',
          '## Matched transfer controls', '',
          'Each trainable arm uses 500,224 tokens, the same stream prefix, initialization,',
          'sites, R=1 reader, AdamW and cosine schedule. DISABLED is the unchanged stock',
          'baseline. Lower full-val NLL gives REAL > PERMUTED > RANDOM > DISABLED.', '',
          '| Control | Full-val NLL | General | Code | Math | Scientific | Multilingual |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for name in ('DISABLED', 'RANDOM', 'PERMUTED', 'REAL'):
    lines.append('| %s | %.6f | %s |' % (name, c['full_nll'][name],
                 ' | '.join('%.6f' % c['domains'][name][d] for d in domains)))
lines += ['', '| Full-val comparison | Paired delta [95% CI] |', '| --- | --- |']
for name, vals in c['comparisons'].items():
    lines.append('| %s | %s |' % (name, ci(vals['full'])))
lines += ['', 'All six full-val intervals exclude zero in the expected direction. REAL also',
          'beats PERMUTED and RANDOM in every domain with intervals below zero. All',
          'domain paired comparisons are in `control-controls-500k-summary.json`.', '',
          '## Execution', '',
          'Kaggle provisioned two separate 16 GiB T4s. All model and reader computation',
          'used GPU0; GPU1 had zero model allocation. Model parallelism was unnecessary',
          'and was not benchmarked. No inter-GPU transfer overhead was incurred.', '',
          'The single-T4 qualification measured 3,196.55 forward tokens/s and 810.19',
          'training tokens/s over short benchmark loops, with 4.309/8.042 GiB forward/',
          'training peak allocation. Actual reader-training peaks were 7.784 GiB.', '',
          '| Reader segment | Tokens | Training tok/s | Training wall seconds |',
          '| --- | ---: | ---: | ---: |']
segments = []
for filename in ('qwengram-2b-reader-0-5m-t4.log', 'qwengram-2b-reader-5-10m-t4.log',
                 'qwengram-2b-reader-10-15m-t4.log'):
    start = None
    previous = 0 if '0-5m' in filename else (5000192 if '5-10m' in filename else 10000384)
    for e in read(filename):
        data = e['data']
        if data.startswith('frozen training stream SHA verified'):
            start = e['time']
        if data.startswith('READER ') and start is not None:
            seen = int(data.split()[2]); seconds = e['time'] - start
            segments.append({'start_tokens': previous, 'end_tokens': seen,
                             'wall_s': seconds, 'tok_s': (seen-previous)/seconds})
            lines.append('| %s→%s | %d | %.1f | %.1f |' %
                         (previous, seen, seen-previous, (seen-previous)/seconds, seconds))
            previous = seen
lines += ['', 'Training intervals come from timestamped logs, ending at reader-weight save',
          'and excluding validation. The original 1M→5M summary divided cumulative',
          'tokens by suffix time; the table corrects that reporting error. Raw logs remain intact.',
          '', 'Host RSS peaked at 13,560 MiB in reader training. Compact suffix caches used',
          '5.701, 5.754 and 5.760 GiB; the calibration/evaluation cache used 2.199 GiB.',
          'Linear750 trained 749,568 tokens in %.1f s (%.1f tokens/s), with %.3f GiB peak GPU allocation.' %
          (s['arb_time_s'], s['arb_training_tok_s'], s['arb_peak_gib']), '',
          'Successful reader, arbitration/evaluation and control kernels took %.3f hours total;' %
          (runtime['total_kernel_wall_seconds']/3600),
          'including the successful qualification benchmark gives %.3f hours.' %
          (runtime['total_with_benchmark_seconds']/3600),
          'These are summed kernel wall times, including setup/cache/evaluation. They exclude',
          'queueing, dataset uploads and interruption-related idle time. Exact environment and',
          'per-kernel durations are in `runtime.json`.', '', '## Persistent artifacts', '',
          'Private dataset: `ninnix/qwengram-2b-checkpoints`. `milestone-shas.json` records',
          'all checkpoint and result SHA256 values. `persistence-verification.json` records',
          'the complete download-back verification. The 5M/10M/15M recovery checkpoints,',
          'linear750 state, controls, per-item evaluations, bootstrap data and frozen stream',
          'are retained. Compact manifests retain row and global-address-map hashes; caches',
          'can be rebuilt exactly from the immutable mounted master PLE.']
(R / 'metrics.md').write_text('\n'.join(lines) + '\n')
runtime['reader_training_intervals'] = segments
(R / 'runtime.json').write_text(json.dumps(runtime, indent=2))

frozen = json.loads((ROOT / 'frozen.json').read_text())
definition = {'name': 'Qwengram-2B', 'recipe': 'REAL-15M + linear750',
              'backbone': frozen['target_model'], 'backbone_revision': frozen['target_revision'],
              'tokenizer_sha256': frozen['target_tokenizer_sha256'],
              'ple_model': frozen['source_model'], 'ple_revision': frozen['source_revision'],
              'ple_manifest_sha256': frozen['ple_manifest_sha256'],
              'reader_tokens': 15000064, 'calibration_tokens': 749568,
              'reader_branches': 1, 'hidden_size': 2048,
              'injection_sites': frozen['injection_sites'],
              'early_alpha_bounds': [0.75, 1.75], 'late_alpha_bounds': [0.0, 0.5],
              'early_alpha': s['early_alpha'], 'gammas': s['gammas'],
              'private_dataset': 'ninnix/qwengram-2b-checkpoints',
              'reader_file': 'reader-15000064.safetensors',
              'reader_sha256': s['reader_sha256'],
              'arbiter_file': 'arb-linear-749568.pt',
              'arbiter_sha256': s['arb_checkpoint_sha256'],
              'metrics': 'results/metrics.md', 'decision': 'results/decision.md',
              'status': 'transfer verified; experimental runtime release justified'}
(ROOT / 'qwengram-2b.json').write_text(json.dumps(definition, indent=2))
