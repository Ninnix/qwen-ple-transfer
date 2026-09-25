import json

import modal


TARGET_ID = "Qwen/Qwen3.6-35B-A3B"
SOURCE_ID = "Qwen/Qwen3.8-Flash-Next-FP8"

app = modal.App("qwen36-ple-source-inspect")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers==5.16.1",
    "huggingface_hub==1.30.0",
)


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=20 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def inspect_source():
    import os
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    revisions = {model: api.model_info(model).sha for model in (TARGET_ID, SOURCE_ID)}
    configs = {
        model: AutoConfig.from_pretrained(model, token=token, trust_remote_code=True)
        for model in (TARGET_ID, SOURCE_ID)
    }
    tokenizers = {
        model: AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=True)
        for model in (TARGET_ID, SOURCE_ID)
    }
    tokenizer_paths = {
        model: hf_hub_download(model, "tokenizer.json", token=token, cache_dir="/data/hf/hub")
        for model in (TARGET_ID, SOURCE_ID)
    }
    tokenizer_hashes = {model: sha256(path) for model, path in tokenizer_paths.items()}
    tokenizer_json = {
        model: json.loads(Path(path).read_text(encoding="utf-8"))
        for model, path in tokenizer_paths.items()
    }
    target_tokenizer = tokenizers[TARGET_ID]
    source_tokenizer = tokenizers[SOURCE_ID]
    vocab_equal = target_tokenizer.get_vocab() == source_tokenizer.get_vocab()
    special_equal = target_tokenizer.special_tokens_map == source_tokenizer.special_tokens_map
    added_equal = target_tokenizer.get_added_vocab() == source_tokenizer.get_added_vocab()
    tokenizer_components = {}
    for name in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        tokenizer_components[name] = tokenizer_json[TARGET_ID].get(name) == tokenizer_json[SOURCE_ID].get(name)
    tokenizer_equal = vocab_equal and special_equal and added_equal and all(tokenizer_components.values())
    if not tokenizer_equal:
        raise RuntimeError(
            f"Tokenizer mismatch: vocab={vocab_equal}, special={special_equal}, added={added_equal}"
        )

    index_path = hf_hub_download(
        SOURCE_ID,
        "model.safetensors.index.json",
        token=token,
        cache_dir="/data/hf/hub",
    )
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    ple_weights = {
        name: shard
        for name, shard in index["weight_map"].items()
        if "ple" in name.lower() or "ngram" in name.lower()
    }
    ple_shards = sorted(set(ple_weights.values()))
    source_files = api.list_repo_files(SOURCE_ID)
    code_files = sorted(
        name
        for name in source_files
        if name.endswith(".py") or "config" in name.lower() or "token" in name.lower()
    )
    downloaded_code = []
    for name in code_files:
        if name.endswith(".py"):
            downloaded_code.append(
                hf_hub_download(SOURCE_ID, name, token=token, cache_dir="/data/hf/hub")
            )

    source_config = configs[SOURCE_ID]
    text_config = getattr(source_config, "text_config", source_config)
    metadata = {
        "source_model": SOURCE_ID,
        "source_revision": revisions[SOURCE_ID],
        "target_model": TARGET_ID,
        "target_revision": revisions[TARGET_ID],
        "tokenizer": {
            "target_sha256": tokenizer_hashes[TARGET_ID],
            "source_sha256": tokenizer_hashes[SOURCE_ID],
            "byte_equal": tokenizer_hashes[TARGET_ID] == tokenizer_hashes[SOURCE_ID],
            "vocabulary_equal": vocab_equal,
            "special_tokens_equal": special_equal,
            "added_tokens_equal": added_equal,
            "semantic_equal": tokenizer_equal,
            "components_equal": tokenizer_components,
        },
        "ple_config": {
            name: getattr(text_config, name, None)
            for name in (
                "hidden_size",
                "ple_embed_dim",
                "ngram_size",
                "heads_per_ngram",
                "ple_layer_ids",
                "split_ngram_parts",
                "ngram_vocab_size_base",
                "vocab_size",
            )
        },
        "ple_tensors": ple_weights,
        "ple_shards": ple_shards,
        "repository_code_files": code_files,
        "downloaded_code_files": [str(Path(path).name) for path in downloaded_code],
        "checkpoint_total_size": index.get("metadata", {}).get("total_size"),
    }
    output = Path("/data/eval/source_ple_metadata.json")
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    volume.commit()
    return metadata


@app.local_entrypoint()
def main():
    print(json.dumps(inspect_source.remote(), indent=2))
