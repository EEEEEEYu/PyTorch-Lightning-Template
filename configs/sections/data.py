from dataclasses import dataclass, field
from typing import Dict, Optional

from configs.config_tracker import TrackedConfigMixin


@dataclass
class AugmentationConfig(TrackedConfigMixin):
    enabled: bool = False
    probability: float = 0.0


@dataclass
class DatasetConfig(TrackedConfigMixin):
    file_name: str = "cifar10"
    class_name: str = "Cifar10"
    root: str = "dataset"
    image_height: int = 32
    image_width: int = 32
    num_classes: int = 10
    augmentation_cfg: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class DataloaderConfig(TrackedConfigMixin):
    batch_size: int = 32
    test_batch_size: Optional[int] = None
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    multiprocessing_context: str = "fork"
    drop_last: bool = False
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False


@dataclass
class DataConfig(TrackedConfigMixin):
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
