import json

import modal


app = modal.App("qwen36-ple-io-benchmark")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0",
    "huggingface_hub==1.30.0",
    "safetensors==0.8.0",
).add_local_dir("qwen36-35b/src", remote_path="/root/src")


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=20 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def benchmark():
    import sys
    import time
    from pathlib import Path

    import torch
    sys.path.insert(0, "/root/src")
    from qwen36_ple.ple_store import MMapPLEStore

    manifest_path = "/data/ple/qwen3.8-flash-next-fp8/manifest.json"
    store = MMapPLEStore.from_manifest(manifest_path)
    generator = torch.Generator().manual_seed(1234)
    results = []
    checksums = []
    for iteration in range(5):
        indices = torch.randint(0, store.rows_per_part * 128, (1, 512, 16), generator=generator)
        started = time.perf_counter()
        output = store.lookup(indices)
        seconds = time.perf_counter() - started
        checksum = output.float().sum().item()
        checksums.append(checksum)
        results.append(
            {
                "iteration": iteration,
                "seconds": round(seconds, 6),
                "tokens_per_second": round(512 / seconds, 2),
                "rows_per_second": round(indices.numel() / seconds, 2),
                "checksum": checksum,
            }
        )
    result = {
        "status": "passed",
        "scope": "128 logical parts",
        "tokens": 512,
        "rows": indices.numel(),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "iterations": results,
    }
    if len(set(checksums)) != len(checksums):
        raise RuntimeError("Benchmark did not use distinct lookup sets")
    Path("/data/eval/ple_io.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(benchmark.remote(), indent=2))
