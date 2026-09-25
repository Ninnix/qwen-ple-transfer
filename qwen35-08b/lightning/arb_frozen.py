# Frozen arbitration math, verbatim from qwen35-08b/tools/arb_stage_src.py.
# Only w, b, alpha2_raw are parameters (~1K). Gate feature uses h8.detach().
# IDX2: alpha2 in [0.75, 1.75], init 1.25. IDX8: alpha8_t = 0.5*sigmoid(w.RMSNorm(h8)+b).
import math
import torch
from torch import nn
ALPHA2_LO = 0.75
ALPHA2_SPAN = 1.0
B_INIT = -1.0986122886681098
def rms_norm(x, eps=1e-6):
    return x.float().mul(torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)
class MemoryArbitration(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(hidden))
        self.b = nn.Parameter(torch.tensor(float(B_INIT)))
        self.alpha2_raw = nn.Parameter(torch.tensor(0.0))
    def alpha2(self):
        return ALPHA2_LO + ALPHA2_SPAN * torch.sigmoid(self.alpha2_raw)
    def alpha8(self, h8):
        hn = rms_norm(h8.detach().float())
        logit = (hn * self.w.float()).sum(-1) + self.b.float()
        return 0.5 * torch.sigmoid(logit)
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
class ArbHooks(nn.Module):
    # Frozen readers give aug = h + gamma*o, so delta = alpha*(aug-h) applies the
    # effective scale without touching gamma. Only arb params receive gradients.
    def __init__(self, model, frozen, arb, decoder_fn):
        super().__init__()
        self.arb = arb
        self.fr2 = frozen.readers["2"]
        self.fr8 = frozen.readers["8"]
        self.fixed_alpha2 = None
        self.memory = None
        self.cur_alpha8 = None
        self.last_alpha8 = None
        dec = decoder_fn(model)
        self.handles = [
            dec[2].register_forward_pre_hook(self._hook2(), with_kwargs=True),
            dec[8].register_forward_pre_hook(self._hook8(), with_kwargs=True),
        ]
    def _a2(self):
        if self.fixed_alpha2 is not None:
            return torch.tensor(float(self.fixed_alpha2), device=self.arb.alpha2_raw.device)
        return self.arb.alpha2()
    def _hook2(self):
        def fn(mod, args, kw):
            assert self.memory is not None
            h = args[0] if args else kw["hidden_states"]
            m = self.memory
            if m.shape[1] != h.shape[1]:
                m = m[:, :h.shape[1]]
            aug = self.fr2(h, m)
            a2 = self._a2().to(h.dtype)
            out = h + a2 * (aug - h)
            if args:
                return (out, *args[1:]), kw
            kw["hidden_states"] = out
            return args, kw
        return fn
    def _hook8(self):
        def fn(mod, args, kw):
            assert self.memory is not None
            h = args[0] if args else kw["hidden_states"]
            m = self.memory
            if m.shape[1] != h.shape[1]:
                m = m[:, :h.shape[1]]
            a8 = self.arb.alpha8(h)
            self.cur_alpha8 = a8
            self.last_alpha8 = a8.detach()
            aug = self.fr8(h, m)
            w8 = a8.to(h.dtype).unsqueeze(-1)
            out = h + w8 * (aug - h)
            if args:
                return (out, *args[1:]), kw
            kw["hidden_states"] = out
            return args, kw
        return fn
    def set_memory(self, m):
        self.memory = m
    def close(self):
        [h.remove() for h in self.handles]
        self.handles.clear()
