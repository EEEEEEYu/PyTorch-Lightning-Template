from dataclasses import dataclass
from typing import Any, Optional

from configs.config_tracker import TrackedConfigMixin


@dataclass
class DistributedConfig(TrackedConfigMixin):
    accelerator: str = "auto"
    devices: Any = 1
    num_nodes: int = 1
    strategy: Optional[str] = None
