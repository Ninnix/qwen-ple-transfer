import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
arms = tuple(summary)
keys = ('mean', 'std', 'p05', 'p50', 'p95')

def items(rows, dataset):
    if dataset == 'hellaswag':
        assert len(rows) == 4000
        return [torch.cat(rows[i:i+4]).numpy() for i in range(0, len(rows), 4)]
    return [row.numpy() for row in rows]

def stats(rows):
    out = np.empty((len(rows), 5), dtype=np.float64)
    for i, row in enumerate(rows):
        x = np.asarray(row, dtype=np.float64)
        out[i] = x.mean(), x.std(), *np.quantile(x, (0.05, 0.5, 0.95))
    return out

per = {}
for arm in arms:
    raw = torch.load(root / ('alpha8-%s.pt' % arm), map_location='cpu', weights_only=False)
    per[arm] = {}
    for dataset, rows in raw.items():
        chunks = items(rows, dataset)
        per[arm][dataset] = stats(chunks)
        all_values = np.concatenate(chunks).astype(np.float64)
        got = (all_values.mean(), all_values.std(), *np.quantile(all_values, (0.05, 0.5, 0.95)))
        ref = summary[arm]['alpha8'][dataset]
        assert all(abs(value - ref[key]) < 2e-5 for key,value in zip(keys, got)), (arm, dataset)
    print(arm, 'raw alpha8 verified', flush=True)

pairs = [('reader7p5m-arb500', 'reader5m-linear750'),
         ('reader7p5m-arb750', 'reader5m-linear750'),
         ('reader10m-arb500', 'reader5m-linear750'),
         ('reader10m-arb750', 'reader5m-linear750'),
         ('reader7p5m-arb750', 'reader7p5m-arb500'),
         ('reader10m-arb750', 'reader10m-arb500'),
         ('reader10m-arb750', 'reader7p5m-arb750')]
report = {}
for a,b in pairs:
    name = a + '-vs-' + b
    report[name] = {}
    for dataset in per[a]:
        delta = per[a][dataset] - per[b][dataset]
        n = len(delta)
        rng = np.random.default_rng(1234)
        picks = rng.integers(n, size=(10000, n), dtype=np.int32)
        report[name][dataset] = {}
        for j,key in enumerate(keys):
            samples = np.sort(delta[picks,j].mean(axis=1))
            p = min(1.0, 2*min(np.count_nonzero(samples>=0),np.count_nonzero(samples<=0))/len(samples))
            report[name][dataset][key] = {
                'global_delta':summary[a]['alpha8'][dataset][key]-summary[b]['alpha8'][dataset][key],
                'paired_item_mean_delta':float(delta[:,j].mean()),
                'ci95':[float(samples[250]),float(samples[9749])],
                'p':float(p), 'n':n, 'paired_item_deltas':delta[:,j].tolist(),
            }
        if dataset in ('full-val','lambada'):
            d = report[name][dataset]['mean']
            print(name,dataset,'alpha8 mean %+0.6f [%+0.6f, %+0.6f]' % (d['paired_item_mean_delta'],*d['ci95']),flush=True)

(root / 'alpha8-paired.json').write_text(json.dumps(report, indent=2))
