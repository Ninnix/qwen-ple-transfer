import json

import modal


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

app = modal.App("qwen36-ple-reader-test")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0",
    "transformers==5.16.1",
    "accelerate==1.14.0",
    "bitsandbytes==0.50.2",
    "huggingface_hub==1.30.0",
    "safetensors==0.8.0",
).add_local_dir("qwen36-35b/src", remote_path="/root/src")


def gpu_memory(torch):
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
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
def reader_test():
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, "/root/src")
    from qwen36_ple import ReaderConfig, ReaderInjection

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

    sequence_length = 32
    prompt = tokenizer("Frozen backbone reader proof. ", return_tensors="pt").input_ids[0]
    repeats = (sequence_length + prompt.numel() - 1) // prompt.numel()
    input_ids = prompt.repeat(repeats)[:sequence_length].unsqueeze(0).to("cuda")
    memory = torch.randn(1, sequence_length, 2560, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        baseline = model(input_ids=input_ids, use_cache=False).logits

    reader_config = ReaderConfig(injection_layers=(2,), branches=1, gamma=0.0)
    injection = ReaderInjection(model, reader_config).to(device="cuda", dtype=torch.bfloat16)
    injection.set_memory(memory)
    trainable_names = [name for name, parameter in injection.named_parameters() if parameter.requires_grad]
    backbone_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if backbone_trainable:
        raise RuntimeError(f"Backbone parameters are trainable: {backbone_trainable[:5]}")

    with torch.inference_mode():
        identity = model(input_ids=input_ids, use_cache=False).logits
    identity_max_abs_diff = (baseline - identity).abs().max().item()
    if not torch.equal(baseline, identity):
        raise RuntimeError(f"Identity test failed: max abs diff {identity_max_abs_diff}")
    del baseline, identity

    reader = injection.readers["2"]
    with torch.no_grad():
        reader.gamma.fill_(1e-3)
    optimizer = torch.optim.AdamW(injection.parameters(), lr=3e-5, weight_decay=0.01)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    reader_ids = {id(parameter) for parameter in injection.parameters()}
    if optimizer_ids != reader_ids:
        raise RuntimeError("Optimizer parameters do not exactly match reader parameters")

    reader_before = {name: parameter.detach().clone() for name, parameter in injection.named_parameters()}
    backbone_sample = model.model.layers[2].input_layernorm.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits = model(input_ids=input_ids, use_cache=False).logits
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
    loss.backward()
    gradient_names = [name for name, parameter in injection.named_parameters() if parameter.grad is not None]
    missing_gradients = [name for name, parameter in injection.named_parameters() if parameter.grad is None]
    backbone_gradients = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if missing_gradients or backbone_gradients:
        raise RuntimeError(f"Gradient audit failed: missing={missing_gradients}, backbone={backbone_gradients[:5]}")
    torch.nn.utils.clip_grad_norm_(injection.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started

    changed = [
        name
        for name, parameter in injection.named_parameters()
        if not torch.equal(reader_before[name], parameter.detach())
    ]
    if not changed:
        raise RuntimeError("No reader parameter changed after optimizer step")
    backbone_unchanged = torch.equal(backbone_sample, model.model.layers[2].input_layernorm.weight.detach())
    if not backbone_unchanged:
        raise RuntimeError("Sampled backbone weight changed")

    result = {
        "status": "passed",
        "sequence_length": sequence_length,
        "reader_parameters": sum(parameter.numel() for parameter in injection.parameters()),
        "trainable_parameters": trainable_names,
        "identity_max_abs_diff": identity_max_abs_diff,
        "loss": loss.item(),
        "reader_gradients": gradient_names,
        "backbone_gradient_count": len(backbone_gradients),
        "changed_reader_parameters": changed,
        "backbone_sample_unchanged": backbone_unchanged,
        "gate_stats": reader.last_gate_stats,
        "gamma": reader.gamma.item(),
        "step_seconds": round(step_seconds, 3),
        "memory": gpu_memory(torch),
    }
    output = Path("/data/eval/reader_proof.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    volume.commit()
    injection.close()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(reader_test.remote(), indent=2))
