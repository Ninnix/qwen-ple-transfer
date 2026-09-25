from torch import nn

from .config import ReaderConfig
from .reader import SharedValueReader


class ReaderInjection(nn.Module):
    def __init__(self, model, config: ReaderConfig):
        super().__init__()
        self.readers = nn.ModuleDict({str(layer): SharedValueReader(config) for layer in config.injection_layers})
        self.memory = None
        self.handles = []
        for layer in config.injection_layers:
            handle = model.model.layers[layer].register_forward_pre_hook(self._hook(str(layer)), with_kwargs=True)
            self.handles.append(handle)

    def _hook(self, name):
        def inject(module, args, kwargs):
            if self.memory is None:
                return args, kwargs
            if args:
                return (self.readers[name](args[0], self.memory), *args[1:]), kwargs
            kwargs["hidden_states"] = self.readers[name](kwargs["hidden_states"], self.memory)
            return args, kwargs

        return inject

    def set_memory(self, memory):
        self.memory = memory

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
