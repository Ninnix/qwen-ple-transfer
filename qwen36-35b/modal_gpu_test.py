import json

import modal


app = modal.App("qwen36-ple-gpu-test")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0")


@app.function(image=image, gpu="A100-80GB", timeout=10 * 60)
def gpu_test():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if "A100" not in properties.name or total_gib < 75:
        raise RuntimeError(f"Expected A100 80GB, got {properties.name} with {total_gib:.2f} GiB")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")

    left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    result = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("BF16 matrix multiplication produced non-finite values")

    return {
        "gpu": properties.name,
        "vram_gib": round(total_gib, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": str(torch.__version__),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "matmul_shape": list(result.shape),
        "matmul_dtype": str(result.dtype),
        "status": "passed",
    }


@app.local_entrypoint()
def main():
    print(json.dumps(gpu_test.remote(), indent=2))
