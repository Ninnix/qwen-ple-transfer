import hashlib
import json
import math
import random
import re
from pathlib import Path


pattern = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$")
here = Path(__file__).resolve().parent
logs = {
    "stock_bf16": here / "logs/stock-bf16.log",
    "qwengram_bf16": here / "logs/qwengram-bf16.log",
    "stock_q8": here / "logs/stock-q8.log",
    "qwengram_q8": here / "logs/qwengram-q8.log",
    "stock_q4": here / "logs/stock-q4.log",
    "qwengram_q4": here / "logs/qwengram-q4.log",
}


def read(path):
    rows = [m.groups() for line in Path(path).read_text().splitlines() if (m := pattern.match(line))]
    assert len(rows) == 64, (path, len(rows))
    assert [int(row[0]) for row in rows] == list(range(0, 64 * 256, 256))
    cumulative = [float(row[2]) for row in rows]
    chunks = [i * cumulative[i - 1] - (i - 1) * (cumulative[i - 2] if i > 1 else 0) for i in range(1, 65)]
    return cumulative[-1], chunks


scores = {name: read(path) for name, path in logs.items()}
gains = {}
for quant in ("bf16", "q8", "q4"):
    stock, qwengram = scores["stock_" + quant], scores["qwengram_" + quant]
    gains[quant] = [a - b for a, b in zip(stock[1], qwengram[1])]

rng = random.Random(1234)
replicates = {name: [] for name in ("bf16", "q8", "q4", "q8_retention", "q4_retention", "q8_minus_bf16", "q4_minus_bf16")}
for _ in range(10000):
    indices = [rng.randrange(16) for _ in range(16)]
    sample = {quant: sum(sum(gains[quant][4 * i:4 * i + 4]) for i in indices) / 64 for quant in gains}
    for quant in gains:
        replicates[quant].append(sample[quant])
    for quant in ("q8", "q4"):
        replicates[quant + "_retention"].append(sample[quant] / sample["bf16"])
        replicates[quant + "_minus_bf16"].append(sample[quant] - sample["bf16"])

intervals = {name: [sorted(values)[250], sorted(values)[9750]] for name, values in replicates.items()}
mean = {name: score[0] for name, score in scores.items()}
gain = {quant: mean["stock_" + quant] - mean["qwengram_" + quant] for quant in gains}
verification = json.loads((here / "verification.json").read_text())
source = json.loads((here / "source.json").read_text())
result = {
    "dataset": "ggml-org/ci wikitext-2-raw-v1.zip: wiki.test.raw",
    "dataset_sha256": "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
    "dataset_zip_sha256": "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
    "method": "first 64 consecutive 256-token chunks; last 127 tokens scored per chunk; CPU; paired bootstrap of 16 consecutive four-chunk blocks, 10000 resamples, seed 1234",
    "scored_tokens": 64 * 127,
    "source": source,
    "canonical": json.loads((here.parent / "qwengram-2b.json").read_text()),
    "ple_sidecar": json.loads((here / "sidecar-validation.json").read_text()),
    "verification": verification,
    "log_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in logs.items()},
    "model_nll": mean,
    "model_ppl": {name: math.exp(value) for name, value in mean.items()},
    "reader_gain_nll": gain,
    "reader_gain_95ci": {quant: intervals[quant] for quant in gains},
    "perplexity_reduction_percent": {quant: 100 * (1 - math.exp(-gain[quant])) for quant in gains},
    "retention": {quant: gain[quant] / gain["bf16"] for quant in ("q8", "q4")},
    "retention_95ci": {quant: intervals[quant + "_retention"] for quant in ("q8", "q4")},
    "gain_difference_vs_bf16": {quant: gain[quant] - gain["bf16"] for quant in ("q8", "q4")},
    "gain_difference_95ci": {quant: intervals[quant + "_minus_bf16"] for quant in ("q8", "q4")},
    "chunk_nll": {name: values[1] for name, values in scores.items()},
}
(here / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print("reader_gain_nll", gain)
print("retention", result["retention"])
print("retention_95ci", result["retention_95ci"])
