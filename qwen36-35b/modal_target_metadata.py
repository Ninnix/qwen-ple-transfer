import json

import modal


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

app = modal.App("qwen36-ple-target-metadata")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers==5.16.1",
    "huggingface_hub==1.30.0",
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def target_metadata():
    import json
    import os
    from collections import Counter
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    config = AutoConfig.from_pretrained(MODEL_ID, token=token, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    files = HfApi(token=token).list_repo_files(MODEL_ID)
    index_path = hf_hub_download(
        MODEL_ID,
        "model.safetensors.index.json",
        token=token,
        cache_dir="/data/hf/hub",
    )
    weight_map = json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"]
    key_roots = Counter(".".join(name.split(".")[:3]) for name in weight_map)
    vision_shards = sorted({shard for name, shard in weight_map.items() if "visual" in name})
    text_shards = sorted({shard for name, shard in weight_map.items() if "language_model" in name})
    result = {
        "status": "passed",
        "model_id": MODEL_ID,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", []) or [],
        "text_config_class": text_config.__class__.__name__,
        "text_model_type": getattr(text_config, "model_type", None),
        "text_architectures": getattr(text_config, "architectures", []) or [],
        "hidden_size": getattr(text_config, "hidden_size", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "num_experts": getattr(text_config, "num_experts", None),
        "num_experts_per_tok": getattr(text_config, "num_experts_per_tok", None),
        "vocab_size": getattr(text_config, "vocab_size", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_size": len(tokenizer),
        "vision_config_fields": [
            name for name in ("vision_config", "visual", "vision_model") if hasattr(config, name)
        ],
        "index_files": [name for name in files if name.endswith(".safetensors.index.json")],
        "safetensors_count": sum(name.endswith(".safetensors") for name in files),
        "checkpoint_key_roots": dict(key_roots.most_common(12)),
        "vision_shards": vision_shards,
        "text_shards": text_shards,
        "shared_vision_text_shards": sorted(set(vision_shards) & set(text_shards)),
    }
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(target_metadata.remote(), indent=2))
