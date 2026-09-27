import time
import torch


class ContiguousT4:
    def __init__(self, model, split=16):
        assert torch.cuda.device_count() == 2
        assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in (0, 1))
        assert len(model.model.layers) == 32 and split == 16
        assert model.lm_head.weight is model.model.embed_tokens.weight
        self.model = model
        self.profile = False
        self.transfers = []
        self.cache = {}
        self.handles = []
        for layer in model.model.layers[split:]:
            layer.to('cuda:1')
            self.handles.append(layer.register_forward_pre_hook(self.layer_input, with_kwargs=True))
        model.model.norm.to('cuda:1')
        self.handles.append(model.register_forward_pre_hook(self.begin, with_kwargs=True))
        self.handles.append(model.lm_head.register_forward_pre_hook(self.head_input))
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert model.model.embed_tokens.weight.device == torch.device('cuda:0')
        assert all(next(layer.parameters()).device.index == (i >= split)
                   for i, layer in enumerate(model.model.layers))
        assert all(not p.requires_grad for p in model.parameters())
        torch.cuda.empty_cache()

    def begin(self, module, args, kwargs):
        assert kwargs.get('use_cache') is False
        self.cache.clear()

    def measured_copy(self, tensor, device, label):
        for i in (0, 1): torch.cuda.synchronize(i)
        start = time.perf_counter()
        out = tensor.to(device)
        torch.cuda.synchronize(device)
        self.transfers.append({'label':label, 'source':str(tensor.device), 'target':str(device),
                               'bytes':tensor.numel()*tensor.element_size(),
                               'seconds':time.perf_counter()-start})
        return out

    def move(self, tensor, device, label):
        if tensor.device == device:
            return tensor
        if not self.profile:
            return tensor.to(device)
        owner = self

        class ProfileCopy(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.source = value.device
                return owner.measured_copy(value, device, label + ':forward')

            @staticmethod
            def backward(ctx, grad):
                return owner.measured_copy(grad, ctx.source, label + ':backward')

        return ProfileCopy.apply(tensor)

    def metadata(self, value):
        if isinstance(value, torch.Tensor):
            if value.device == torch.device('cuda:1'):
                return value
            assert not value.requires_grad
            key = id(value)
            if key not in self.cache:
                self.cache[key] = (value, self.move(value, torch.device('cuda:1'), 'metadata'))
            return self.cache[key][1]
        if isinstance(value, tuple): return tuple(self.metadata(v) for v in value)
        if isinstance(value, list): return [self.metadata(v) for v in value]
        if isinstance(value, dict): return {k:self.metadata(v) for k,v in value.items()}
        return value

    def layer_input(self, module, args, kwargs):
        assert kwargs.get('past_key_values') is None
        assert len(args) == 1
        hidden = self.move(args[0], torch.device('cuda:1'), 'decoder-boundary')
        return (hidden,), self.metadata(kwargs)

    def head_input(self, module, args):
        assert len(args) == 1
        return (self.move(args[0], torch.device('cuda:0'), 'tied-output-head'),)

    def description(self):
        return {'strategy':'contiguous-T4x2', 'decoder_gpu0':[0,15], 'decoder_gpu1':[16,31],
                'reader_devices':{'3':0,'11':0}, 'embedding_and_tied_head_gpu':0,
                'final_norm_gpu':1, 'hidden_transfers_per_forward':2,
                'shared_metadata_copied_once_per_forward':True,
                'tied_embedding_preserved':True}
