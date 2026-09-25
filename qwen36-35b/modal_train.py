import json
import math

import modal


MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SEQUENCE_LENGTH = 512
VALIDATION_DIR = "/data/datasets/fineweb-edu-validation-v1"
PLE_MANIFEST = "/data/ple/qwen3.8-flash-next-fp8/manifest.json"
CHECKPOINT_TOKENS = (100000, 250000, 500000)

app = modal.App("qwen36-ple-train")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.8.0",
    "transformers==5.16.1",
    "accelerate==1.14.0",
    "bitsandbytes==0.50.2",
    "huggingface_hub==1.30.0",
    "safetensors==0.8.0",
    "datasets==4.0.0",
).add_local_dir("qwen36-35b/src", remote_path="/root/src")


def token_stream(dataset, tokenizer):
    for row in dataset:
        text = row.get("text")
        if not text:
            continue
        yield from tokenizer.encode(text, add_special_tokens=False)
        yield tokenizer.eos_token_id


def read_validation(torch):
    import hashlib
    import sys
    from array import array
    from pathlib import Path

    directory = Path(VALIDATION_DIR)
    metadata = json.loads((directory / "validation.json").read_text(encoding="ascii"))
    data = (directory / "tokens.uint32le").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != metadata["tokens_sha256"]:
        raise RuntimeError(f"Validation checksum mismatch: {digest}")
    tokens = array("I")
    tokens.frombytes(data)
    if sys.byteorder != "little":
        tokens.byteswap()
    tensor = torch.tensor(tokens, dtype=torch.long).view(-1, SEQUENCE_LENGTH)
    if tensor.numel() != metadata["token_count"]:
        raise RuntimeError("Validation token count mismatch")
    return tensor, metadata


def evaluate(model, injection, store, validation, torch, use_reader):
    import torch.nn.functional as F

    total_loss = 0.0
    total_targets = 0
    gates = {name: [] for name in injection.readers}
    with torch.inference_mode():
        for block in validation:
            input_ids_cpu = block.unsqueeze(0)
            if use_reader:
                memory = store.lookup(__import__("qwen36_ple").ngram_indices(input_ids_cpu)).to("cuda")
                injection.set_memory(memory)
            else:
                injection.set_memory(None)
            input_ids = input_ids_cpu.to("cuda")
            logits = model(input_ids=input_ids, use_cache=False).logits
            targets = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                targets,
                reduction="sum",
            )
            total_loss += loss.item()
            total_targets += targets.numel()
            if use_reader:
                for name, reader in injection.readers.items():
                    gates[name].append(reader.last_gate.float().cpu().reshape(-1))
            del logits, input_ids
    loss = total_loss / total_targets
    result = {"loss": loss, "perplexity": math.exp(min(loss, 20))}
    if use_reader:
        by_layer = {}
        all_gates = []
        for name, values in gates.items():
            gate = torch.cat(values)
            all_gates.append(gate)
            quantiles = torch.quantile(gate, torch.tensor([0.05, 0.5, 0.95]))
            by_layer[name] = {
                "mean": gate.mean().item(),
                "std": gate.std(correction=0).item(),
                "p05": quantiles[0].item(),
                "p50": quantiles[1].item(),
                "p95": quantiles[2].item(),
            }
        gate = torch.cat(all_gates)
        quantiles = torch.quantile(gate, torch.tensor([0.05, 0.5, 0.95]))
        result["gate"] = {
            "mean": gate.mean().item(),
            "std": gate.std(correction=0).item(),
            "p05": quantiles[0].item(),
            "p50": quantiles[1].item(),
            "p95": quantiles[2].item(),
        }
        result["gate_by_layer"] = by_layer
    return result


def update_norm(injection, initial, suffix, torch):
    total = 0.0
    for name, parameter in injection.named_parameters():
        if name.endswith(suffix):
            delta = parameter.detach().float().cpu() - initial[name].float()
            total += delta.square().sum().item()
    return math.sqrt(total)


def save_checkpoint(output, injection, optimizer, scheduler, reader_metadata, metrics, run, torch):
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in injection.state_dict().items()}
    save_file(state, output / "reader.safetensors")
    (output / "reader.json").write_text(json.dumps(reader_metadata, indent=2) + "\n", encoding="ascii")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="ascii")
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="ascii")
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        output / "optimizer.pt",
    )
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=131072,
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("qwen36-ple-hf")],
    volumes={"/data": volume},
)
def train(max_tokens: int, placement: str = "2", seed: int = 1234):
    import os
    import random
    import sys
    import time
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if max_tokens != 500000:
        raise ValueError("This controlled placement run requires --max-tokens 500000")
    placements = {"2": (2,), "13": (13,), "2-13": (2, 13)}
    if placement not in placements:
        raise ValueError("--placement must be 2, 13, or 2-13")
    injection_layers = placements[placement]
    sys.path.insert(0, "/root/src")
    from qwen36_ple import MMapPLEStore, ReaderConfig, ReaderInjection, ngram_indices

    os.environ["HF_HOME"] = "/data/hf"
    os.environ["HF_HUB_CACHE"] = "/data/hf/hub"
    os.environ["HF_DATASETS_CACHE"] = "/data/datasets/cache"
    token = os.environ["HF_TOKEN"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    validation, validation_metadata = read_validation(torch)
    api = HfApi(token=token)
    target_revision = api.model_info(MODEL_ID).sha
    dataset_revision = api.dataset_info(DATASET_ID).sha
    if target_revision != validation_metadata["target_revision"]:
        raise RuntimeError("Target revision differs from frozen validation metadata")
    if dataset_revision != validation_metadata["dataset_revision"]:
        raise RuntimeError("Dataset revision differs from frozen validation metadata")

    config = AutoConfig.from_pretrained(MODEL_ID, token=token, revision=target_revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, revision=target_revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=target_revision,
        config=config.text_config,
        token=token,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)

    reader_config = ReaderConfig(injection_layers=injection_layers, branches=1, gamma=1e-3)
    injection = ReaderInjection(model, reader_config).to(device="cuda", dtype=torch.bfloat16)
    trainable = [(name, parameter) for name, parameter in injection.named_parameters() if parameter.requires_grad]
    trainable_ids = {id(parameter) for _, parameter in trainable}
    backbone_ids = {id(parameter) for parameter in model.parameters()}
    if trainable_ids & backbone_ids or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Reader/backbone trainability boundary is invalid")

    step_lengths = []
    seen = 0
    for checkpoint in CHECKPOINT_TOKENS:
        while seen < checkpoint:
            length = min(SEQUENCE_LENGTH, checkpoint - seen)
            step_lengths.append(length)
            seen += length
    total_steps = len(step_lengths)
    warmup_steps = min(total_steps, max(10, math.ceil(total_steps * 0.05)))
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=3e-5, weight_decay=0.01)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("Optimizer does not contain exactly the reader parameters")

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    store = MMapPLEStore.from_manifest(PLE_MANIFEST)
    ple_stats = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(set(store.part_paths.values()))
    }
    backbone_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    initial_reader = {name: parameter.detach().cpu().clone() for name, parameter in injection.named_parameters()}

    run_root = Path(f"/data/checkpoints/layer{placement}-r1-500k")
    completed_tokens = 0
    checkpoint_results = []
    for checkpoint_tokens in CHECKPOINT_TOKENS:
        checkpoint_dir = run_root / f"checkpoint-{checkpoint_tokens}"
        if not (checkpoint_dir / "reader.safetensors").exists():
            break
        completed_tokens = checkpoint_tokens
        checkpoint_results.append(
            json.loads((checkpoint_dir / "metrics.json").read_text(encoding="ascii"))
        )
    if completed_tokens == max_tokens:
        result_path = Path(f"/data/eval/layer{placement}-r1-500k.json")
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="ascii"))
        raise RuntimeError("Final checkpoint exists but aggregate evaluation result is missing")
    if completed_tokens:
        from safetensors.torch import load_file

        checkpoint_dir = run_root / f"checkpoint-{completed_tokens}"
        injection.load_state_dict(load_file(checkpoint_dir / "reader.safetensors"))
        optimizer_state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(optimizer_state["optimizer"])
        scheduler.load_state_dict(optimizer_state["scheduler"])

    plan = {
        "gpu": torch.cuda.get_device_name(0),
        "target_model": MODEL_ID,
        "target_precision": "bitsandbytes-int8",
        "reader_sites": list(injection_layers),
        "reader_branches": 1,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_names": [name for name, _ in trainable],
        "sequence_length": SEQUENCE_LENGTH,
        "microbatch": 1,
        "gradient_accumulation": 1,
        "token_budget": max_tokens,
        "optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "warmup_fraction": warmup_steps / total_steps,
        "validation_tokens": validation.numel(),
        "validation_sha256": validation_metadata["tokens_sha256"],
        "training_document_skip": validation_metadata["training_starts_at_document"],
        "resumed_from_tokens": completed_tokens,
    }
    print(json.dumps(plan, indent=2), flush=True)

    baseline_path = Path("/data/eval/fixed_validation_baseline.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="ascii"))
        if baseline["validation_sha256"] != validation_metadata["tokens_sha256"]:
            raise RuntimeError("Cached baseline uses a different validation set")
    else:
        baseline = evaluate(model, injection, store, validation, torch, use_reader=False)
        baseline.update(
            {
                "target_model": MODEL_ID,
                "target_revision": target_revision,
                "target_precision": "bitsandbytes-int8",
                "validation_sha256": validation_metadata["tokens_sha256"],
                "validation_tokens": validation.numel(),
            }
        )
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="ascii")
        volume.commit()
    print(json.dumps({"baseline_validation": baseline}, indent=2), flush=True)

    dataset = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=dataset_revision,
    ).skip(validation_metadata["training_starts_at_document"])
    stream = token_stream(dataset, tokenizer)
    for _ in range(completed_tokens):
        next(stream)
    reader_metadata = {
        "format": "qwen-ple-reader",
        "version": 1,
        "source_model": "Qwen/Qwen3.8-Flash-Next-FP8",
        "target_model": MODEL_ID,
        "memory_dim": 2560,
        "hidden_dim": 2048,
        "injection_layers": list(injection_layers),
        "branches": 1,
        "addressing": "qwen4-exp-splitmix64-xor-v1",
        "tokenizer_sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        "ple_revision": "236dfdf285828023ca3bcd3f37366c58a3469b13",
        "target_revision": target_revision,
    }
    run = {
        **plan,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "gradient_clip": 1.0,
        "target_revision": target_revision,
        "ple_revision": reader_metadata["ple_revision"],
    }

    started = time.perf_counter()
    interval_loss_sum = 0.0
    interval_targets = 0
    tokens_seen = completed_tokens
    last_gradient_norm = None
    torch.cuda.reset_peak_memory_stats()
    completed_steps = next(
        (index for index, total in enumerate(__import__("itertools").accumulate(step_lengths), start=1) if total == completed_tokens),
        0,
    )
    for step, current_length in enumerate(step_lengths[completed_steps:], start=completed_steps + 1):
        token_ids = [next(stream) for _ in range(current_length)]
        input_ids_cpu = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        memory = store.lookup(ngram_indices(input_ids_cpu)).to("cuda")
        injection.set_memory(memory)
        input_ids = input_ids_cpu.to("cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, use_cache=False).logits
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError(f"Backbone gradient detected at step {step}")
        last_gradient_norm = torch.nn.utils.clip_grad_norm_(injection.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        tokens_seen += current_length
        interval_loss_sum += loss.item() * targets.numel()
        interval_targets += targets.numel()
        del logits, loss, input_ids, memory

        if tokens_seen in CHECKPOINT_TOKENS:
            changed_versions = [
                name for name, parameter in model.named_parameters() if parameter._version != backbone_versions[name]
            ]
            if changed_versions:
                raise RuntimeError(f"Backbone weights changed: {changed_versions[:5]}")
            current_ple_stats = {
                str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in sorted(set(store.part_paths.values()))
            }
            if current_ple_stats != ple_stats:
                raise RuntimeError("PLE files changed during training")
            validation_result = evaluate(model, injection, store, validation, torch, use_reader=True)
            checkpoint = {
                "tokens": tokens_seen,
                "step": step,
                "training_loss": interval_loss_sum / interval_targets,
                "validation_loss": validation_result["loss"],
                "validation_perplexity": validation_result["perplexity"],
                "validation_delta_vs_baseline": validation_result["loss"] - baseline["loss"],
                "gamma": {name: parameter.item() for name, parameter in injection.named_parameters() if name.endswith("gamma")},
                "gate": validation_result["gate"],
                "gate_by_layer": validation_result["gate_by_layer"],
                "reader_gradient_norm": last_gradient_norm,
                "w_k_update_norm": update_norm(injection, initial_reader, "keys.0.weight", torch),
                "w_v_update_norm": update_norm(injection, initial_reader, "value.weight", torch),
                "beta_update_norm": update_norm(injection, initial_reader, "beta", torch),
                "learning_rate": scheduler.get_last_lr()[0],
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "tokens_per_second": round((tokens_seen - completed_tokens) / (time.perf_counter() - started), 2),
                "backbone_gradient_count": 0,
                "backbone_changed_parameter_count": 0,
                "ple_files_unchanged": True,
                "optimizer_reader_only": optimizer_ids == trainable_ids,
            }
            checkpoint_results.append(checkpoint)
            output = Path(f"/data/checkpoints/layer{placement}-r1-500k/checkpoint-{tokens_seen}")
            save_checkpoint(output, injection, optimizer, scheduler, reader_metadata, checkpoint, run, torch)
            print(json.dumps(checkpoint), flush=True)
            interval_loss_sum = 0.0
            interval_targets = 0

    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "plan": plan,
        "baseline_validation": baseline,
        "checkpoints": checkpoint_results,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    result_path = Path(f"/data/eval/layer{placement}-r1-500k.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    volume.commit()
    injection.close()
    return result


@app.local_entrypoint()
def main(max_tokens: int = 0, placement: str = "", seed: int = 1234):
    if max_tokens != 500000:
        raise ValueError("Pass --max-tokens 500000 for the controlled placement run")
    if placement not in {"2", "13", "2-13"}:
        raise ValueError("Pass --placement 2, 13, or 2-13")
    print(
        json.dumps(
            {"token_budget": max_tokens, "placement": placement, "checkpoints": CHECKPOINT_TOKENS},
            indent=2,
        )
    )
    print(json.dumps(train.remote(max_tokens, placement, seed), indent=2))
