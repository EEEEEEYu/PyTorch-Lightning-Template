from dataclasses import dataclass

from configs.config_tracker import TrackedConfigMixin

@dataclass
class CheckpointConfig(TrackedConfigMixin):
    enabled: bool = True
    every_n_epochs: int = 1
    monitor: str = "val_loss"
    mode: str = "min"
    filename: str = "best-{epoch:03d}-{val_loss:.5f}"
    save_top_k: int = 1
    save_last: bool = True
