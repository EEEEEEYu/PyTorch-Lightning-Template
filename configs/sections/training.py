from dataclasses import dataclass

from configs.config_tracker import TrackedConfigMixin


@dataclass
class TrainingConfig(TrackedConfigMixin):
    deterministic: bool = False
    use_compile: bool = False
    inference_mode: bool = False
    seed: int = 42
    max_epochs: int = 1
