import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
arms = {'canonical319488':'canonical', '500736':'500', '749568':'750', '1001472':'1000'}
raw = {arm:torch.load(root / 'milestones' / ('alpha8-%s.pt' % tag), map_location='cpu', weights_only=False) for arm, tag in arms.items()}
keys = ('mean', 'std', 'p05', 'p50', 'p95')

def units(rows, dataset):
    if dataset == 'hellaswag':
        assert len(rows) == 4000
        return [torch.cat(rows[i:i+4]).numpy() for i in range(0, len(rows), 4)]
    return [x.numpy() for x in rows]

def stats(items):
    out = np.empty((len(items), 5), dtype=np.float64)
    for i, item in enumerate(items):
        x = np.asarray(item, dtype=np.float64)
        q = np.quantile(x, (0.05, 0.5, 0.95))
        out[i] = x.mean(), x.std(), *q
    return out

per = {}
for arm, datasets in raw.items():
    per[arm] = {}
    for dataset, rows in datasets.items():
        items = units(rows, dataset)
        per[arm][dataset] = stats(items)
        all_alpha = np.concatenate(items).astype(np.float64)
        got = (all_alpha.mean(), all_alpha.std(), *np.quantile(all_alpha, (0.05, 0.5, 0.95)))
        ref = summary['canonical' if arm == 'canonical319488' else arm]['alpha8'][dataset]
        assert all(abs(value - ref[key]) < 2e-5 for key, value in zip(keys, got)), (arm, dataset)
        print('%s %s raw alpha8 matches saved global stats, %d items' % (arm, dataset, len(items)))

pairs = [('500736', 'canonical319488'), ('749568', '500736'),
         ('1001472', '749568'), ('1001472', 'canonical319488')]
report = {}
for a, b in pairs:
    name = a + '-vs-' + b
    report[name] = {}
    for dataset in per[a]:
        aa, bb = per[a][dataset], per[b][dataset]
        assert aa.shape == bb.shape
        delta = aa - bb
        n = len(delta)
        rng = np.random.default_rng(1234)
        picks = rng.integers(n, size=(10000, n), dtype=np.int32)
        av = summary['canonical' if a == 'canonical319488' else a]['alpha8'][dataset]
        bv = summary['canonical' if b == 'canonical319488' else b]['alpha8'][dataset]
        report[name][dataset] = {}
        for j, key in enumerate(keys):
            samples = np.sort(delta[picks, j].mean(axis=1))
            p = min(1.0, 2.0 * min(np.count_nonzero(samples >= 0), np.count_nonzero(samples <= 0)) / len(samples))
            report[name][dataset][key] = {
                'global_delta': av[key] - bv[key],
                'paired_item_mean_delta': float(delta[:, j].mean()),
                'ci95': [float(samples[250]), float(samples[9749])],
                'p': float(p), 'n': n,
                'paired_item_deltas': delta[:, j].tolist(),
            }
        if dataset in ('full-val', 'lambada'):
            print(name, dataset, 'alpha8 mean %+0.6f [%+0.6f, %+0.6f]' %
                  (report[name][dataset]['mean']['paired_item_mean_delta'],
                   *report[name][dataset]['mean']['ci95']))

(root / 'alpha8-paired.json').write_text(json.dumps(report, indent=2))
print('wrote', root / 'alpha8-paired.json')
