import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
SPEC = importlib.util.spec_from_file_location('reference_compare', HERE.parent / 'qwen35-2b/milestone-compare/analyze.py')
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(calibrated):
    reasons = []
    for key in ('val_nll', 'domain_mean_nll'):
        if calibrated[key]['hi'] >= 0:
            reasons.append(key + ': 15M improvement unresolved')
    for key, metric in {**calibrated['domains'], 'lambada_nll': calibrated['lambada_nll']}.items():
        if metric['lo'] > 0:
            reasons.append(key + ': statistically clear 15M NLL regression')
    for key in ('hs_acc', 'lambada_acc'):
        if calibrated[key]['hi'] < 0:
            reasons.append(key + ': statistically clear 15M accuracy regression')
    return (10 if reasons else 15), reasons


def ci(metric, factor=1):
    return '%+.5f [%+.5f, %+.5f]' % tuple(factor * metric[k] for k in ('mean', 'lo', 'hi'))


def reduction(nll, stock):
    return 100 * (1 - math.exp(nll - stock))


def main():
    inputs = {}
    manifest = json.loads((HERE / '.artifacts/checkpoints/milestone-shas.json').read_text())

    def load(name):
        path = RESULTS / name
        assert sha(path) == manifest[name]['sha256'], name
        inputs[name] = sha(path)
        return json.loads(path.read_text())

    stock = load('arb10-eval-stock.json')
    repeat = load('arb15-eval-stock.json')
    tol = json.loads((HERE / 'protocol.json').read_text())['qualification']['stock_repeat_nll_tolerance']
    for key in ('hs_correct', 'lam_correct'):
        assert stock[key] == repeat[key], key
    differences = {'val': max(abs(x['sum']/x['n']-y['sum']/y['n']) for x,y in zip(stock['val_blocks'],repeat['val_blocks'])),
                   'lambada': max(abs(x-y) for x,y in zip(stock['lam_nlls'],repeat['lam_nlls']))}
    for domain in stock['domains']:
        differences[domain] = max(abs(x['sum']/x['n']-y['sum']/y['n'])
                                  for x,y in zip(stock['dom_blocks'][domain],repeat['dom_blocks'][domain]))
    assert max(differences.values()) <= tol, differences
    evals = {'stock': stock}
    summaries, bootstraps = {}, {}
    for stage in (10, 15):
        evals['raw%d' % stage] = load('arb%d-eval-raw-%dm.json' % (stage, stage))
        evals['linear%d' % stage] = load('arb%d-eval-linear750.json' % stage)
        summaries[stage] = load('arb%d-summary.json' % stage)
        bootstraps[stage] = load('arb%d-bootstrap.json' % stage)
        assert summaries[stage]['eval_streams_sha256'] == sha(RESULTS / 'evaluation-streams.json')
    pairs = {mode: REFERENCE.contrast(evals[mode+'15'], evals[mode+'10'], mode+' 15M - 10M')
             for mode in ('raw', 'linear')}
    endpoint, reasons = select(pairs['linear'])
    canonical = evals['linear%d' % endpoint]
    controls = load('control-controls-500k-summary.json')
    disabled = load('control-control-eval-DISABLED.json')
    assert max(abs(x['sum']/x['n']-y['sum']/y['n']) for x,y in zip(stock['val_blocks'],disabled['full'])) <= tol
    transfer = all(controls['comparisons']['REAL_minus_'+name]['full']['hi'] < 0
                   for name in ('PERMUTED','RANDOM'))
    stock_ci = bootstraps[endpoint]['linear_minus_stock']
    stock_gain = stock_ci['val_nll']['hi'] < 0
    domains_better = [d for d in stock['domains'] if canonical['domains'][d] < stock['domains'][d]]
    clear_domains = [d for d in stock['domains'] if stock_ci['domains'][d]['hi'] < 0]
    regression = [k for k in ('hs_acc','lambada_acc') if stock_ci[k]['hi'] < 0]
    release = transfer and stock_gain and len(domains_better) == 5 and not regression
    reader_stages = {i: load('reader-%dm-summary.json' % i) for i in (5,10,15)}
    bench = load('benchmark-4b.json')
    peaks = bench['gpu_peaks_gib']
    if bench['execution']['strategy'] == 'single-T4-checkpointed':
        execution_answer = ('single T4 qualified with decoder activation checkpointing: '
                            '%.3f GiB peak, %.1f forward tok/s, %.1f training tok/s. '
                            'Ordinary single-T4 backward had run out of memory.') % (
                                peaks[0],bench['forward_tok_s'],bench['training_tok_s'])
    else:
        execution_answer = ('single T4 failed on the first backward. Contiguous T4×2 qualified: '
                            'GPU0 %.3f / GPU1 %.3f GiB peak, %.1f forward tok/s, %.1f training tok/s.') % (
                                *peaks,bench['forward_tok_s'],bench['training_tok_s'])
    perf = reader_stages[endpoint]['reader']
    historical = json.loads((RESULTS / 'cross-scale-reference.json').read_text())
    cross = {k: historical[k] for k in ('0.8B','2B')}
    cross['4B'] = {'stock':stock, 'canonical':canonical,
                   'canonical_reader_tokens':{10:10000384,15:15000064}[endpoint],
                   'raw_nll':evals['raw%d'%endpoint]['val_nll'],
                   'ppl_reduction_pct':reduction(canonical['val_nll'],stock['val_nll']),
                   'raw_ppl_reduction_pct':reduction(evals['raw%d'%endpoint]['val_nll'],stock['val_nll']),
                   'alpha_early':summaries[endpoint]['early_alpha'],
                   'alpha_late':canonical['alpha8']['full-val'],
                   'training':{'tok_s':perf['tok_s'],'peak_gib':perf['peak_GiB'],
                               'gpu_peaks_gib':perf['gpu_peaks_gib']},
                   'control_order_full_val':' > '.join(sorted(controls['full_nll'],key=controls['full_nll'].get))}
    result = {'protocol_sha256':sha(HERE / 'protocol.json'), 'input_shas':inputs,
              'stock_reproduction_max_difference':differences,
              'metrics':{k:REFERENCE.fields(v) for k,v in evals.items()},
              'paired_15m_minus_10m':pairs, 'canonical_reader_m':endpoint,
              'selection_reasons':reasons, 'clear_real_vs_permuted_and_random':transfer,
              'release_followup_justified':release, 'reference_sha256':sha(RESULTS / 'cross-scale-reference.json')}
    (RESULTS / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    order = ('stock','raw10','linear10','raw15','linear15')
    lines = ['# Qwengram-4B decision', '', '**Balanced canonical: REAL-%dM + linear750.**'%endpoint, '',
             'Both raw endpoints and the independently calibrated 10M/15M endpoints are preserved. '
             'The backbone and PLE remained frozen; no architecture search was run.', '',
             '## Decision answers', '',
             '1. **REAL PLE transfer:** %s. The causal controls %s resolve a REAL advantage over both PERMUTED and RANDOM.' %
             ('supported' if transfer and stock_gain else 'not established by all required tests', 'do' if transfer else 'do not'),
             '2. **Controls:** %s (full-validation point ordering; paired intervals below).' % cross['4B']['control_order_full_val'],
             '3. **Raw perplexity reduction:** 10M %.3f%%; 15M %.3f%%.' % tuple(reduction(evals['raw%d'%i]['val_nll'],stock['val_nll']) for i in (10,15)),
             '4. **Calibrated perplexity reduction:** 10M %.3f%%; 15M %.3f%%.' % tuple(reduction(evals['linear%d'%i]['val_nll'],stock['val_nll']) for i in (10,15)),
             '5. **Canonical endpoint:** %dM under the frozen balanced rule. %s' % (endpoint, '; '.join(reasons) if reasons else '15M clears both aggregate CIs without a statistically clear domain or benchmark regression.'),
             '6. **Domains:** %d/5 improve by point estimate; %d/5 have paired NLL intervals entirely below stock.' % (len(domains_better),len(clear_domains)),
             '7. **Benchmarks versus stock:** HellaSwag %s pp; LAMBADA accuracy %s pp; LAMBADA NLL %s.' %
             (ci(stock_ci['hs_acc'],100),ci(stock_ci['lambada_acc'],100),ci(stock_ci['lambada_nll'])),
             '8. **Cross-scale calibrated gain:** 0.8B %.3f%%; 2B %.3f%%; 4B %.3f%%.' % tuple(cross[k]['ppl_reduction_pct'] for k in ('0.8B','2B','4B')),
             '9. **Execution:** ' + execution_answer,
             '10. **GGUF/runtime follow-up:** %s. No export or further experiment was started.' %
             ('justified for a subsequent matched runtime validation; these results do not validate a GGUF' if release else 'not justified as a balanced release by the present evidence'), '',
             '## Frozen evaluation', '',
             '| Arm | Full NLL | Δ NLL | PPL | PPL reduction | Domain mean | HS % | LAMBADA % | LAMBADA NLL |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in order:
        e=evals[key]
        lines.append('| %s | %.6f | %+.6f | %.6f | %.3f%% | %.6f | %.2f | %.2f | %.6f |' %
                     (key,e['val_nll'],e['val_nll']-stock['val_nll'],math.exp(e['val_nll']),reduction(e['val_nll'],stock['val_nll']),e['domain_mean'],100*e['hs_acc'],100*e['lambada_acc'],e['lambada_nll']))
    lines += ['', '| Domain | Stock | Raw10 | Linear10 | Raw15 | Linear15 | Canonical − stock [95% CI] |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for d in stock['domains']:
        lines.append('| %s | %s | %s |' % (d,' | '.join('%.6f'%evals[k]['domains'][d] for k in order),ci(stock_ci['domains'][d])))
    lines += ['', '## Mandatory paired 15M minus 10M', '',
              'Negative NLL favors 15M; positive accuracy favors 15M. Accuracy differences are percentage points.', '',
              '| Metric | Raw delta [95% CI] | Calibrated delta [95% CI] |','| --- | --- | --- |']
    rows = {k:REFERENCE.ci_rows(v) for k,v in pairs.items()}
    for key in rows['raw']:
        factor=100 if key.endswith('_acc') else 1
        lines.append('| %s | %s | %s |' % (key,ci(rows['raw'][key],factor),ci(rows['linear'][key],factor)))
    lines += ['', 'The accuracy regression rule requires the paired interval to be entirely below zero. '
              'Unresolved differences do not establish equivalence. The historical 2B canonical selection '
              'used its earlier, more conservative rule and is retained independently.', '',
              '## Arbitration', '', '| Endpoint | Early α | Late mean | Std | p05 | p50 | p95 | γ early | γ late | Effective early | Effective late mean |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage in (10,15):
        s=summaries[stage];a=s['linear750']['alpha8']['full-val']
        values=[s['early_alpha'],*(a[k] for k in ('mean','std','p05','p50','p95')),s['gammas']['3'],s['gammas']['11'],s['effective_early'],s['effective_late_mean']]
        lines.append('| %dM | %s |'%(stage,' | '.join('%.6f'%v for v in values)))
    lines += ['', '| Calibration − raw | Full NLL [95% CI] | Domain mean [95% CI] | HS pp [95% CI] | LAMBADA NLL [95% CI] |',
              '| --- | --- | --- | --- | --- |']
    for stage in (10,15):
        c=bootstraps[stage]['linear_minus_raw']
        lines.append('| %dM | %s | %s | %s | %s |'%(stage,ci(c['val_nll']),ci(c['domain_mean_nll']),ci(c['hs_acc'],100),ci(c['lambada_nll'])))
    lines += ['', 'Raw reader multipliers are 1 at both sites; stock injection is disabled. '
              'Per-domain alpha distributions are preserved in the endpoint summaries.', '',
              '| Reader | IDX | W_K update norm | W_V update norm | Beta update norm | Residual norm | Calibrated residual norm | Relative calibrated norm |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage,s in summaries.items():
        for site in ('3','11'):
            p=s['reader_parameter_update_norms'];r=s['reader_residual_update_norms'][site]
            vals=[p['readers.'+site+'.keys.0.weight'],p['readers.'+site+'.value.weight'],p['readers.'+site+'.beta'],r['reader'],r['effective'],r['relative']]
            lines.append('| %dM | %s | %s |'%(stage,site,' | '.join('%.6f'%v for v in vals)))
    lines += ['',
              '## Matched 500K controls', '', '| Condition | Full NLL |','| --- | ---: |']
    for k,v in controls['full_nll'].items(): lines.append('| %s | %.6f |'%(k,v))
    lines += ['', '| Paired comparison | Full-val Δ NLL [95% CI] |','| --- | --- |']
    for k,c in controls['comparisons'].items():
        m=c['full'];lines.append('| %s | %+.6f [%+.6f, %+.6f] |'%(k,m['delta'],m['lo'],m['hi']))
    lines += ['', 'All six paired comparisons for every held-out domain are preserved in `results/control-controls-500k-summary.json`.', '',
              '## Cross-scale comparison', '', '| Scale | Stock NLL / PPL | Canonical NLL / PPL | Δ NLL | PPL reduction | Raw reduction | Reader tokens | HS Δ pp | LAMBADA Δ NLL | Domain mean Δ |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k,e in cross.items():
        b=e['stock'];c=e['canonical']
        lines.append('| %s | %.6f / %.4f | %.6f / %.4f | %+.6f | %.3f%% | %.3f%% | %d | %+.2f | %+.6f | %+.6f |'%
                     (k,b['val_nll'],math.exp(b['val_nll']),c['val_nll'],math.exp(c['val_nll']),c['val_nll']-b['val_nll'],e['ppl_reduction_pct'],e['raw_ppl_reduction_pct'],e['canonical_reader_tokens'],100*(c['hs_acc']-b['hs_acc']),c['lambada_nll']-b['lambada_nll'],c['domain_mean']-b['domain_mean']))
    decreasing=cross['0.8B']['ppl_reduction_pct']>cross['2B']['ppl_reduction_pct']>cross['4B']['ppl_reduction_pct']
    lines += ['', 'The measured relative benefit %s decrease monotonically across these canonical checkpoints. '
              'This is a descriptive comparison of the established tracks, not an isolated causal effect of backbone size.'%('does' if decreasing else 'does not'), '',
              '| Scale | Control order | Early α | Late mean / std / p05 / p50 / p95 | Train tok/s | Peak GiB |',
              '| --- | --- | ---: | --- | ---: | ---: |']
    for k,e in cross.items():
        lines.append('| %s | %s | %.6f | %s | %.1f | %.3f |'%(k,e['control_order_full_val'],e['alpha_early'],' / '.join('%.6f'%e['alpha_late'][x] for x in ('mean','std','p05','p50','p95')),e['training']['tok_s'],e['training']['peak_gib']))
    lines += ['', 'The 0.8B throughput corrects the original cumulative-token numerator to the actual continuation tokens. '
              'Throughput includes milestone validation and is not a matched hardware speed benchmark. '
              'The 2B raw 15M research endpoint separately achieves %.3f%% perplexity reduction.'%historical['2B']['research_raw15_ppl_reduction_pct'], '',
              'The cross-scale peak column is the largest per-device peak; 4B uses %d T4 GPU(s).'%len(peaks), '',
              '## Runtime and artifacts', '', '| Stage | Tokens/s | GPU0 GiB | GPU1 GiB | Host MiB | Wall seconds |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    records=[('reader 0–1M',reader_stages[5]['one_m']),('reader 1–5M',reader_stages[5]['five_m']),
             ('reader 5–10M',reader_stages[10]['reader']),('reader 10–15M',reader_stages[15]['reader'])]
    def memory_columns(values):
        return ' | '.join(['%.3f'%v for v in values]+['—']*(2-len(values)))
    for name,r in records: lines.append('| %s | %.1f | %s | %.1f | %.1f |'%(name,r['tok_s'],memory_columns(r['gpu_peaks_gib']),r['rss_MiB'],r['wall_s']))
    for stage,s in summaries.items(): lines.append('| %dM arbitration | %.1f | %s | %.1f | %.1f |'%(stage,s['arb_training_tok_s'],memory_columns(s['arb_gpu_peaks_gib']),s['host_rss_mib'],s['arb_time_s']))
    transfer = bench['inter_gpu_transfer']
    if transfer:
        lines += ['', 'Qualification measured %.2f ms of synchronized inter-GPU copies per forward/backward '
                  '(%.2f%% of a normal training step, estimated from a separate profiled step). '
                  'Normal throughput excludes profiling synchronizations. Full transfer events and bytes '
                  'are in `results/benchmark-4b.json`.' %
                  (1000*transfer['total_seconds'],100*transfer['fraction_of_normal_step_estimate'])]
    else:
        lines += ['', 'Single-GPU execution has zero inter-GPU transfer overhead. '
                  'Measured training throughput includes decoder recomputation during backward.']
    execution = {job:load(job+'-execution.json')['kernel_elapsed_s'] for job in
                 ('benchmark','train-5m','train-10m','train-15m','arb-10m','arb-15m','controls')}
    interrupted = sorted(RESULTS.glob('interrupted-run*.json'))
    if interrupted:
        for path in interrupted:
            recovery = load(path.name)
            execution['interrupted_%s_observed' % recovery['session_id']] = recovery['observed_execution_seconds']
        lines += ['', 'Recorded Kaggle execution wall time: at least %.2f hours, including observed interrupted attempts, installation, cache construction, training and evaluation. Gaps between the last observed output and confirmation of session cancellation are not measured. Queue and local transfer time are excluded. Recovery restored the verified 1,000,448-token checkpoint with its optimizer, scheduler and RNG state.' % (sum(execution.values())/3600)]
    else:
        lines += ['', 'Total Kaggle execution wall time: %.2f hours, including installation, cache construction, training and evaluation; excluding queue and local upload/download time.' % (sum(execution.values())/3600)]
    lines += ['', 'Training suffix cache sizes: '+', '.join('%dM %.3f GiB'%(stage,load('compact-reader-%d-%dm.json'%(stage-5,stage))['address_count']*164/1024**3) for stage in (5,10,15))+'.', '',
              'The full evaluation uses 1,024 blocks, five domains of 64 blocks, and 1,000 examples per benchmark. '
              'CIs use the original 10,000 paired bootstrap resamples, seed 1234; domain mean is stratified by domain. '
              'Intervals are marginal without multiplicity correction. Repeated stock predictions passed the 2e-6 NLL tolerance and identical accuracy-array checks.', '',
              'Verified checkpoints, raw per-item/per-block evaluations, bootstrap outputs and SHAs are persisted in the private Kaggle checkpoint dataset. '
              'See `results/comparison.json`, the endpoint summaries, `protocol.json`, `frozen.json`, and the runtime manifests. '
              'No further experiment follows this report.', '']
    result['input_shas'] = inputs
    result['kernel_execution_seconds'] = execution
    (RESULTS / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    (HERE / 'decision.md').write_text('\n'.join(lines))
    readme = HERE / 'README.md'
    if readme.exists():
        text = readme.read_text()
        complete = 'The measured study is complete. See [the decision report](decision.md) for the endpoint selection and transfer evidence.'
        for pending in ('GPU qualification is pending; no 4B result or canonical endpoint is claimed.',
                        'Single-T4 qualification passed; reader training is the next stage. No 4B quality result or canonical endpoint is claimed.',
                        'Single-T4 qualification passed; reader training is running. No 4B quality result or canonical endpoint is claimed.'):
            text = text.replace(pending, complete)
        readme.write_text(text)
    print('FINAL_REPORT', endpoint, 'M canonical; release follow-up justified:', release)


if __name__ == '__main__':
    main()
