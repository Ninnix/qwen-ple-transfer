import json

import modal


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

app = modal.App("qwen36-ple-target-test")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0",
    "transformers==5.16.1",
    "accelerate==1.14.0",
    "bitsandbytes==0.50.2",
    "huggingface_hub==1.30.0",
    "safetensors==0.8.0",
)


def memory_stats(torch):
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=131072,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def target_test():
    import os
    import time
    from pathlib import Path

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    config = AutoConfig.from_pretrained(MODEL_ID, token=token, trust_remote_code=True)
    text_config = config.text_config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, trust_remote_code=True)
    quantization = BitsAndBytesConfig(load_in_8bit=True)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=text_config,
        token=token,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=quantization,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    load_seconds = time.perf_counter() - started
    model.eval()
    model.requires_grad_(False)
    after_load = memory_stats(torch)

    model_type = getattr(text_config, "model_type", None)
    visual_modules = [name for name, _ in model.named_modules() if "visual" in name or "vision" in name]
    if visual_modules:
        raise RuntimeError(f"Text-only model contains visual modules: {visual_modules[:5]}")
    missing_keys = sorted(loading_info.get("missing_keys", []))
    if missing_keys:
        raise RuntimeError(f"Text-only load has missing weights: {missing_keys[:5]}")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Frozen target has trainable parameters: {trainable[:5]}")

    prompt_ids = tokenizer("The quick brown fox jumps over the lazy dog. ", return_tensors="pt").input_ids[0]
    forwards = {}
    for sequence_length in (128, 512):
        repeats = (sequence_length + prompt_ids.numel() - 1) // prompt_ids.numel()
        input_ids = prompt_ids.repeat(repeats)[:sequence_length].unsqueeze(0).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if logits.shape != (1, sequence_length, text_config.vocab_size):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError(f"Non-finite logits at sequence length {sequence_length}")
        forwards[str(sequence_length)] = {
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(sequence_length / elapsed, 2),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "memory": memory_stats(torch),
        }
        del logits, input_ids

    result = {
        "status": "passed",
        "model_id": MODEL_ID,
        "model_class": model.__class__.__name__,
        "model_type": model_type,
        "visual_modules": visual_modules,
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(loading_info.get("unexpected_keys", [])),
        "text_only_loader": "AutoModelForCausalLM",
        "quantization": "bitsandbytes-int8",
        "load_seconds": round(load_seconds, 3),
        "model_memory_footprint_gib": round(model.get_memory_footprint() / 1024**3, 3),
        "memory_after_load": after_load,
        "forwards": forwards,
        "cuda_version": torch.version.cuda,
        "pytorch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
    }
    output = Path("/data/eval/target_load.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(target_test.remote(), indent=2))
