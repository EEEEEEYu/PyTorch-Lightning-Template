from dataclasses import dataclass, field
from typing import Any, Dict

from configs.config_tracker import TrackedConfigMixin


@dataclass
class GradientAccumulationConfig(TrackedConfigMixin):
    enabled: bool = False
    scheduling: Dict[int, int] = field(default_factory=lambda: {0: 1})


@dataclass
class GradientClipConfig(TrackedConfigMixin):
    enabled: bool = False
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = "norm"


@dataclass
class SWAConfig(TrackedConfigMixin):
    enabled: bool = False
    swa_lrs: float = 1e-2


@dataclass
class OptimizerConfig(TrackedConfigMixin):
    name: str = "Adam"
    arguments: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})
    gradient_accumulation: GradientAccumulationConfig = field(default_factory=GradientAccumulationConfig)
    gradient_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    stochastic_weight_averaging: SWAConfig = field(default_factory=SWAConfig)
