import json

import modal


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
VALIDATION_TOKENS = 524288

app = modal.App("qwen36-ple-validation-prepare")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers==5.16.1",
    "huggingface_hub==1.30.0",
    "datasets==4.0.0",
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def prepare_validation():
    import hashlib
    import os
    import sys
    from array import array
    from pathlib import Path

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    target_revision = api.model_info(MODEL_ID).sha
    dataset_revision = api.dataset_info(DATASET_ID).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, revision=target_revision)
    dataset = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )

    tokens = array("I")
    documents_consumed = 0
    for row in dataset:
        documents_consumed += 1
        text = row.get("text")
        if text:
            tokens.extend(tokenizer.encode(text, add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= VALIDATION_TOKENS:
            del tokens[VALIDATION_TOKENS:]
            break
    if len(tokens) != VALIDATION_TOKENS:
        raise RuntimeError(f"Expected {VALIDATION_TOKENS} tokens, got {len(tokens)}")
    if sys.byteorder != "little":
        tokens.byteswap()

    directory = Path("/data/datasets/fineweb-edu-validation-v1")
    directory.mkdir(parents=True, exist_ok=True)
    token_path = directory / "tokens.uint32le"
    token_path.write_bytes(tokens.tobytes())
    token_sha256 = hashlib.sha256(token_path.read_bytes()).hexdigest()
    metadata = {
        "format": "qwen-token-validation",
        "version": 1,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "split": "train",
        "documents_consumed": documents_consumed,
        "training_starts_at_document": documents_consumed,
        "token_count": VALIDATION_TOKENS,
        "dtype": "uint32",
        "byte_order": "little",
        "sequence_length": 512,
        "sequence_count": VALIDATION_TOKENS // 512,
        "target_model": MODEL_ID,
        "target_revision": target_revision,
        "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokens_sha256": token_sha256,
    }
    metadata_path = directory / "validation.json"
    existing = json.loads(metadata_path.read_text(encoding="ascii")) if metadata_path.exists() else None
    if existing is not None and existing != metadata:
        raise RuntimeError("Existing frozen validation metadata differs from the regenerated artifact")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    volume.commit()
    return {"status": "passed", **metadata, "path": str(token_path)}


@app.local_entrypoint()
def main():
    print(json.dumps(prepare_validation.remote(), indent=2))
