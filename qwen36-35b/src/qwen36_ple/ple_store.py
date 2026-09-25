from pathlib import Path

import torch
from safetensors import safe_open


class PLEStore:
    def lookup(self, indices):
        raise NotImplementedError


class MMapPLEStore(PLEStore):
    def __init__(self, part_paths, weight_scale, rows_per_part=2_500_012, row_dim=160):
        self.part_paths = {int(part): Path(path) for part, path in part_paths.items()}
        self.weight_scale = torch.as_tensor(weight_scale, dtype=torch.bfloat16).reshape(1)
        self.rows_per_part = rows_per_part
        self.row_dim = row_dim
        self.weights = {}

    @classmethod
    def from_manifest(cls, path):
        import json

        path = Path(path)
        manifest = json.loads(path.read_text(encoding="ascii"))
        part_paths = {part: path.parent / shard for part, shard in manifest["parts"].items()}
        scale_path = path.parent / manifest["parts"]["0"]
        return cls(
            part_paths,
            load_weight_scale(scale_path),
            rows_per_part=manifest["rows_per_part"],
            row_dim=manifest["row_dim"],
        )

    @staticmethod
    def tensor_name(part):
        return (
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
            f"shard_{part}.weight"
        )

    def lookup(self, indices):
        shape = indices.shape
        flat = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        output = torch.empty((flat.numel(), self.row_dim), dtype=torch.bfloat16)
        parts = torch.div(flat, self.rows_per_part, rounding_mode="floor")
        local_rows = torch.remainder(flat, self.rows_per_part)
        for part in torch.unique(parts).tolist():
            path = self.part_paths.get(part)
            if path is None:
                raise FileNotFoundError(f"PLE part {part} is not available")
            positions = torch.nonzero(parts == part, as_tuple=False).flatten()
            weight = self.weights.get(part)
            if weight is None:
                with safe_open(path, framework="pt", device="cpu") as file:
                    weight = file.get_tensor(self.tensor_name(part))
                self.weights[part] = weight
            rows = weight.index_select(0, local_rows.index_select(0, positions))
            output.index_copy_(0, positions, rows.to(torch.bfloat16) * self.weight_scale)
        return output.reshape(*shape, self.row_dim).flatten(-2)


def load_weight_scale(path):
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    with safe_open(path, framework="pt", device="cpu") as file:
        return file.get_tensor(name)
