import math

import torch
from torch import nn

from .config import ReaderConfig


def rms_norm(x, eps=1e-6):
    return x.float().mul(torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


class SharedValueReader(nn.Module):
    def __init__(self, config: ReaderConfig):
        super().__init__()
        self.config = config
        self.keys = nn.ModuleList(
            nn.Linear(config.memory_dim, config.hidden_dim, bias=False) for _ in range(config.branches)
        )
        self.value = nn.Linear(config.memory_dim, config.hidden_dim, bias=False)
        self.beta = nn.Parameter(torch.zeros(config.branches))
        self.gamma = nn.Parameter(torch.tensor(config.gamma))
        self.last_gate = None

    @property
    def last_gate_stats(self):
        if self.last_gate is None:
            return None
        gate = self.last_gate.float()
        quantiles = torch.quantile(gate, torch.tensor([0.05, 0.5, 0.95], device=gate.device))
        return {
            "mean": gate.mean().item(),
            "std": gate.std(correction=0).item(),
            "variance": gate.var(correction=0).item(),
            "p05": quantiles[0].item(),
            "p50": quantiles[1].item(),
            "p95": quantiles[2].item(),
            "near_zero_fraction": (gate < 0.01).float().mean().item(),
        }

    def forward(self, hidden_states, memory):
        if hidden_states.shape[:-1] != memory.shape[:-1]:
            raise ValueError(f"hidden/memory shape mismatch: {hidden_states.shape} vs {memory.shape}")
        query = rms_norm(hidden_states)
        value = self.value(memory)
        gates = []
        for branch, projection in enumerate(self.keys):
            key = rms_norm(projection(memory))
            score = (query.float() * key.float()).sum(dim=-1) / math.sqrt(self.config.hidden_dim)
            gates.append(torch.sigmoid(score + self.beta[branch].float()).to(value.dtype))
        gate = torch.stack(gates, dim=0)
        output = (gate.unsqueeze(-1) * value.unsqueeze(0)).mean(dim=0)
        self.last_gate = gate.detach()
        return hidden_states + self.gamma.to(output.dtype) * output
