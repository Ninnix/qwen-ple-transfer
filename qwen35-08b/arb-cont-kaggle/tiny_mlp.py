import torch
from torch import nn


class ResidualMLPArbitration(nn.Module):
    def __init__(self, w, b, width=32):
        super().__init__()
        assert w.numel() == 1024 and b.numel() == 1
        self.register_buffer('w', w.detach().float().clone())
        self.register_buffer('b', b.detach().float().clone())
        self.register_buffer('alpha2_raw', torch.tensor(0.0))
        self.fc1 = nn.Linear(1024, width)
        self.fc2 = nn.Linear(width, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def alpha8(self, h8):
        h = h8.detach().float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        linear = (h * self.w).sum(-1) + self.b
        delta = self.fc2(torch.nn.functional.silu(self.fc1(h))).squeeze(-1)
        return 0.5 * torch.sigmoid(linear + delta)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
