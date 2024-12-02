from dataclasses import dataclass, field
from typing import Any, Dict

from configs.config_tracker import TrackedConfigMixin

@dataclass
class ModelConfig(TrackedConfigMixin):
    file_name: str = "simple_net"
    class_name: str = "SimpleNet"
    input_meta: Dict[str, Any] = field(default_factory=dict)
    blocks: Dict[str, Any] = field(default_factory=dict)
