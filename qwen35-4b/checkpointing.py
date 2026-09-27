import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedDecoder:
    def __init__(self, layers):
        self.enabled = True
        self.checkpoint_calls = 0
        self.body_calls = 0
        self.originals = []
        for layer in layers:
            assert not getattr(layer, 'gradient_checkpointing', False)
            assert all(not p.requires_grad for p in layer.parameters())
            self.originals.append(layer.forward)
            layer.forward = self.wrap(layer.forward)

    def wrap(self, original):
        def body(*args, **kwargs):
            self.body_calls += 1
            return original(*args, **kwargs)

        def forward(hidden_states, *args, **kwargs):
            assert kwargs.get('past_key_values') is None
            if self.enabled and torch.is_grad_enabled() and hidden_states.requires_grad:
                self.checkpoint_calls += 1
                # Module hooks run outside forward, so injections are not replayed.
                return checkpoint(body, hidden_states, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return original(hidden_states, *args, **kwargs)
        return forward

    def description(self):
        return {'strategy':'single-T4-checkpointed', 'gpu':0,
                'decoder_layers':len(self.originals), 'use_reentrant':False,
                'preserve_rng_state':True, 'injection_hooks_recomputed':False,
                'backbone_eval_mode_preserved':True,
                'checkpoint_calls':self.checkpoint_calls, 'body_calls':self.body_calls}
