import json

import modal


SOURCE_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
PHYSICAL_SHARD = "model-00005-of-00131.safetensors"

app = modal.App("qwen36-ple-row-test")
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
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def ple_test():
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    sys.path.insert(0, "/root/src")
    from qwen36_ple.ple_store import MMapPLEStore, load_weight_scale

    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    started = time.perf_counter()
    path = hf_hub_download(
        SOURCE_ID,
        PHYSICAL_SHARD,
        token=os.environ["HF_TOKEN"],
        cache_dir="/data/hf/hub",
    )
    download_seconds = time.perf_counter() - started
    scale = load_weight_scale(path)
    parts = {}
    headers = {}
    with safe_open(path, framework="pt", device="cpu") as file:
        for part in (0, 1):
            name = MMapPLEStore.tensor_name(part)
            tensor = file.get_tensor(name)
            parts[part] = path
            headers[str(part)] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        scale_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        headers["weight_scale"] = {
            "shape": list(file.get_tensor(scale_name).shape),
            "dtype": str(file.get_tensor(scale_name).dtype),
        }

    store = MMapPLEStore(parts, scale)
    local = torch.tensor(
        [[0, 1, 127, 1024, 2_500_011], [17, 200_003, 999_999, 2_000_000, 2_499_999]],
        dtype=torch.long,
    )
    global_indices = torch.stack((local[0], local[1] + store.rows_per_part))
    actual = store.lookup(global_indices.unsqueeze(-1)).squeeze(-2)

    reference_rows = []
    with safe_open(path, framework="pt", device="cpu") as file:
        for part in (0, 1):
            weight = file.get_tensor(MMapPLEStore.tensor_name(part))
            rows = F.embedding(local[part], weight)
            reference_rows.append(rows.to(scale.dtype) * scale)
    expected = torch.stack(reference_rows)
    max_abs_diff = (actual - expected).abs().max().item()
    if not torch.equal(actual, expected):
        raise RuntimeError(f"FP8 row dequantization mismatch: max abs diff {max_abs_diff}")
    if actual.dtype != torch.bfloat16 or actual.shape != (2, 5, 160):
        raise RuntimeError(f"Unexpected lookup result: {actual.shape} {actual.dtype}")

    result = {
        "status": "passed",
        "physical_shard": PHYSICAL_SHARD,
        "file_size_gib": round(Path(path).stat().st_size / 1024**3, 3),
        "download_seconds": round(download_seconds, 3),
        "headers": headers,
        "weight_scale": scale.item(),
        "lookup_shape": list(actual.shape),
        "lookup_dtype": str(actual.dtype),
        "reference_max_abs_diff": max_abs_diff,
        "sample_checksum": actual.float().sum().item(),
    }
    output = Path("/data/eval/ple_row_test.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(ple_test.remote(), indent=2))
