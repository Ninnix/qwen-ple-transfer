import json
import uuid

import modal


app = modal.App("qwen36-ple-prepare")
volume = modal.Volume.from_name("qwen36-ple-data", create_if_missing=True)
mounts = {"/data": volume}


@app.function(volumes=mounts)
def initialize_volume(marker):
    from pathlib import Path

    directories = ("hf", "ple", "datasets", "checkpoints", "eval", "cache")
    for directory in directories:
        Path("/data", directory).mkdir(parents=True, exist_ok=True)
    Path("/data/cache/persistence-marker.txt").write_text(marker, encoding="ascii")
    volume.commit()
    return {"directories": list(directories), "marker": marker}


@app.function(volumes=mounts)
def verify_volume(marker):
    from pathlib import Path

    volume.reload()
    path = Path("/data/cache/persistence-marker.txt")
    actual = path.read_text(encoding="ascii")
    if actual != marker:
        raise RuntimeError(f"Persistence marker mismatch: expected {marker!r}, got {actual!r}")
    directories = ("hf", "ple", "datasets", "checkpoints", "eval", "cache")
    missing = [name for name in directories if not Path("/data", name).is_dir()]
    if missing:
        raise RuntimeError(f"Missing volume directories: {missing}")
    return {"marker": actual, "status": "passed", "volume": "qwen36-ple-data"}


@app.local_entrypoint()
def main():
    marker = str(uuid.uuid4())
    initialized = initialize_volume.remote(marker)
    verified = verify_volume.remote(marker)
    print(json.dumps({"initialized": initialized, "verified": verified}, indent=2))
