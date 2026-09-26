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
    "stock_q6": here / "logs/stock-q6.log",
    "qwengram_q6": here / "logs/qwengram-q6.log",
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
for quant in ("bf16", "q8", "q6", "q4"):
    stock, qwengram = scores["stock_" + quant], scores["qwengram_" + quant]
    gains[quant] = [a - b for a, b in zip(stock[1], qwengram[1])]

rng = random.Random(1234)
replicates = {name: [] for name in ("bf16", "q8", "q6", "q4", "q8_retention", "q6_retention", "q4_retention", "q8_minus_bf16", "q6_minus_bf16", "q4_minus_bf16")}
for _ in range(10000):
    indices = [rng.randrange(16) for _ in range(16)]
    sample = {quant: sum(sum(gains[quant][4 * i:4 * i + 4]) for i in indices) / 64 for quant in gains}
    for quant in gains:
        replicates[quant].append(sample[quant])
    for quant in ("q8", "q6", "q4"):
        replicates[quant + "_retention"].append(sample[quant] / sample["bf16"])
        replicates[quant + "_minus_bf16"].append(sample[quant] - sample["bf16"])

intervals = {name: [sorted(values)[250], sorted(values)[9750]] for name, values in replicates.items()}
mean = {name: score[0] for name, score in scores.items()}
gain = {quant: mean["stock_" + quant] - mean["qwengram_" + quant] for quant in gains}
result = {
    "dataset": "ggml-org/ci wikitext-2-raw-v1.zip: wiki.test.raw",
    "dataset_sha256": "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
    "dataset_zip_sha256": "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
    "method": "first 64 consecutive 256-token chunks; last 127 tokens scored per chunk; CPU; paired bootstrap of 16 consecutive four-chunk blocks, 10000 resamples, seed 1234",
    "scored_tokens": 64 * 127,
    "stock_bf16_source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "stock_bf16_source_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "ple_sidecar_sha256": "66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1",
    "runtime_commit": "1c5053f23e99341bee106c1232af90f57ed2cfdd",
    "quantizer": "llama.cpp build 9888 (cb295bf59)",
    "gguf_sha256": {
        "stock_bf16": "b36273a8c2f7115257101eabafc6cc54c3088c970617b3b7654686057613b797",
        "qwengram_bf16": "8eee4d1061fcd9d60f223e50e510183d0818758b381ebf22ca192b0a6271318a",
        "stock_q8": "206dcd4e12f3b846601335fa84716a0ed93c4cf47214fcead29cb492093a1f26",
        "qwengram_q8": "ec5f65a28be248b41f981f1be3074dcb09a52987977bcda8d1c352711f01c043",
        "stock_q4": "7c03ffe5eed6bce4392d68fe93cbffa7338391709bdc4124cc8982d4aeaff05f",
        "qwengram_q4": "77500ea47628c2a40155a4d2b6468a1a4aec4950de3bc717268704e40be52e33",
    },
    "q6_verification": json.loads((here / "q6-verification.json").read_text()),
    "log_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in logs.items()},
    "model_nll": mean,
    "model_ppl": {name: math.exp(value) for name, value in mean.items()},
    "reader_gain_nll": gain,
    "reader_gain_95ci": {quant: intervals[quant] for quant in gains},
    "perplexity_reduction_percent": {quant: 100 * (1 - math.exp(-gain[quant])) for quant in gains},
    "retention": {quant: gain[quant] / gain["bf16"] for quant in ("q8", "q6", "q4")},
    "retention_95ci": {quant: intervals[quant + "_retention"] for quant in ("q8", "q6", "q4")},
    "gain_difference_vs_bf16": {quant: gain[quant] - gain["bf16"] for quant in ("q8", "q6", "q4")},
    "gain_difference_95ci": {quant: intervals[quant + "_minus_bf16"] for quant in ("q8", "q6", "q4")},
    "chunk_nll": {name: values[1] for name, values in scores.items()},
}
(here / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print("reader_gain_nll", gain)
print("retention", result["retention"])
print("retention_95ci", result["retention_95ci"])
