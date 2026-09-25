from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderConfig:
    memory_dim: int = 2560
    hidden_dim: int = 2048
    injection_layers: tuple[int, ...] = (2,)
    branches: int = 1
    gamma: float = 0.0
