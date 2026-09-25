import json

import modal


SOURCE_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
SOURCE_REVISION = "236dfdf285828023ca3bcd3f37366c58a3469b13"

app = modal.App("qwen36-ple-download")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==1.30.0")


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def download_ple():
    import os
    import time
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = os.environ["HF_TOKEN"]
    destination = Path("/data/ple/qwen3.8-flash-next-fp8")
    destination.mkdir(parents=True, exist_ok=True)
    physical_shards = [f"model-{number:05d}-of-00131.safetensors" for number in range(5, 38)]
    metadata_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    started = time.perf_counter()
    snapshot_download(
        SOURCE_ID,
        revision=SOURCE_REVISION,
        allow_patterns=physical_shards + metadata_files,
        local_dir=destination,
        token=token,
        max_workers=8,
    )
    download_seconds = time.perf_counter() - started

    index_path = destination / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    parts = {}
    for name, shard in index["weight_map"].items():
        if name.startswith(prefix + "shard_") and name.endswith(".weight"):
            part = int(name.removeprefix(prefix + "shard_").removesuffix(".weight"))
            parts[str(part)] = shard
    if sorted(map(int, parts)) != list(range(128)):
        raise RuntimeError(f"Expected 128 PLE parts, got {len(parts)}")

    info = HfApi(token=token).model_info(SOURCE_ID, revision=SOURCE_REVISION, files_metadata=True)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    files = []
    for name in physical_shards:
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(path)
        sibling = siblings[name]
        files.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": sibling.lfs.get("sha256") if sibling.lfs else None,
            }
        )

    manifest = {
        "format": "qwen-ple",
        "version": 1,
        "source_model": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "tokenizer_sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        "dtype": "F8_E4M3",
        "scale_dtype": "BF16",
        "scale_tensor": prefix + "weight_scale",
        "weight_scale": 0.00019931793212890625,
        "ngram_orders": [2, 3],
        "heads_per_ngram": 8,
        "head_count": 16,
        "row_dim": 160,
        "memory_dim": 2560,
        "rows_per_part": 2500012,
        "part_count": 128,
        "padded_rows": 320001536,
        "addressing": {
            "name": "qwen4-exp-splitmix64-xor-v1",
            "seed": 1234,
            "ple_layer_index": 0,
            "eos_token_id": 248044,
            "vocab_size": 248320,
            "ngram_vocab_size_base": 20000000,
            "make_vocab_divisible_by": 128,
            "layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
            "head_vocab_sizes": [
                20000003,
                20000023,
                20000033,
                20000047,
                20000059,
                20000063,
                20000069,
                20000077,
                20000081,
                20000093,
                20000107,
                20000147,
                20000153,
                20000159,
                20000161,
                20000171,
            ],
            "head_offsets": [
                0,
                20000003,
                40000026,
                60000059,
                80000106,
                100000165,
                120000228,
                140000297,
                160000374,
                180000455,
                200000548,
                220000655,
                240000802,
                260000955,
                280001114,
                300001275,
            ],
        },
        "parts": parts,
        "files": files,
        "download_seconds": round(download_seconds, 3),
        "total_bytes": sum(item["size"] for item in files),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    volume.commit()
    return {
        "status": "passed",
        "files": len(files),
        "parts": len(parts),
        "total_gib": round(manifest["total_bytes"] / 1024**3, 3),
        "download_seconds": manifest["download_seconds"],
        "manifest": str(destination / "manifest.json"),
    }


@app.local_entrypoint()
def main():
    print(json.dumps(download_ple.remote(), indent=2))
