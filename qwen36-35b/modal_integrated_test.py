import json

import modal


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

app = modal.App("qwen36-ple-integrated-test")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0",
    "transformers==5.16.1",
    "accelerate==1.14.0",
    "bitsandbytes==0.50.2",
    "huggingface_hub==1.30.0",
    "safetensors==0.8.0",
).add_local_dir("qwen36-35b/src", remote_path="/root/src")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=131072,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def integrated_test():
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, "/root/src")
    from qwen36_ple import MMapPLEStore, ReaderConfig, ReaderInjection, ngram_indices

    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    token = os.environ["HF_TOKEN"]
    config = AutoConfig.from_pretrained(MODEL_ID, token=token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config.text_config,
        token=token,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)

    sequence_length = 512
    prompt = tokenizer("Real frozen PLE integrated forward. ", return_tensors="pt").input_ids[0]
    repeats = (sequence_length + prompt.numel() - 1) // prompt.numel()
    input_ids_cpu = prompt.repeat(repeats)[:sequence_length].unsqueeze(0)
    indices = ngram_indices(input_ids_cpu)
    store = MMapPLEStore.from_manifest("/data/ple/qwen3.8-flash-next-fp8/manifest.json")
    lookup_times = []
    memory = None
    for _ in range(3):
        started = time.perf_counter()
        memory = store.lookup(indices)
        lookup_times.append(time.perf_counter() - started)
    if memory.shape != (1, sequence_length, 2560) or memory.dtype != torch.bfloat16:
        raise RuntimeError(f"Unexpected PLE memory: {memory.shape} {memory.dtype}")

    input_ids = input_ids_cpu.to("cuda")
    memory = memory.to("cuda")
    with torch.inference_mode():
        baseline = model(input_ids=input_ids, use_cache=False).logits
    injection = ReaderInjection(model, ReaderConfig(injection_layers=(2,), gamma=0.0)).to(
        device="cuda", dtype=torch.bfloat16
    )
    injection.set_memory(memory)
    with torch.inference_mode():
        identity = model(input_ids=input_ids, use_cache=False).logits
    identity_diff = (baseline - identity).abs().max().item()
    if not torch.equal(baseline, identity):
        raise RuntimeError(f"Real-PLE identity mismatch: {identity_diff}")

    with torch.no_grad():
        injection.readers["2"].gamma.fill_(1e-3)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        augmented = model(input_ids=input_ids, use_cache=False).logits
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    nonzero_diff = (baseline - augmented).abs().max().item()
    if nonzero_diff == 0 or not torch.isfinite(augmented).all().item():
        raise RuntimeError(f"Real-PLE augmentation did not produce finite changed logits: {nonzero_diff}")

    result = {
        "status": "passed",
        "sequence_length": sequence_length,
        "indices_shape": list(indices.shape),
        "memory_shape": list(memory.shape),
        "memory_dtype": str(memory.dtype),
        "lookup_seconds": [round(value, 6) for value in lookup_times],
        "lookup_tokens_per_second": [round(sequence_length / value, 2) for value in lookup_times],
        "identity_max_abs_diff": identity_diff,
        "nonzero_max_abs_diff": nonzero_diff,
        "forward_seconds": round(forward_seconds, 3),
        "forward_tokens_per_second": round(sequence_length / forward_seconds, 2),
        "reader_parameters": sum(parameter.numel() for parameter in injection.parameters()),
        "gate_stats": injection.readers["2"].last_gate_stats,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    Path("/data/eval/integrated_forward.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    volume.commit()
    injection.close()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(integrated_test.remote(), indent=2))
