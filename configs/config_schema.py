from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf

from configs.config_tracker import (
    ConfigUsageTracker,
    TrackedConfigMixin,
    attach_tracker,
)
from configs.sections import (
    TrainingConfig,
    DistributedConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    LoggerConfig,
    CheckpointConfig,
)


if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)


@dataclass
class AppConfig(TrackedConfigMixin):
    TRAINING: TrainingConfig = field(default_factory=TrainingConfig)
    DISTRIBUTED: DistributedConfig = field(default_factory=DistributedConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)
    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    LOGGER: LoggerConfig = field(default_factory=LoggerConfig)
    CHECKPOINT: CheckpointConfig = field(default_factory=CheckpointConfig)


def _validate_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def validate_app_config(cfg: AppConfig):
    errors: List[str] = []

    if not cfg.MODEL.file_name:
        errors.append("MODEL.file_name must be specified.")
    if not cfg.MODEL.class_name:
        errors.append("MODEL.class_name must be specified.")

    dataset = cfg.DATA.dataset
    if not dataset.file_name:
        errors.append("DATA.dataset.file_name must be specified.")
    if not dataset.class_name:
        errors.append("DATA.dataset.class_name must be specified.")
    if not _validate_positive(dataset.num_classes):
        errors.append("DATA.dataset.num_classes must be a positive integer.")

    dataloader = cfg.DATA.dataloader
    if not _validate_positive(dataloader.batch_size):
        errors.append("DATA.dataloader.batch_size must be positive.")

    training = cfg.TRAINING
    if not _validate_positive(training.max_epochs):
        errors.append("TRAINING.max_epochs must be positive.")

    optimizer = cfg.OPTIMIZER
    if not optimizer.name:
        errors.append("OPTIMIZER.name must be specified.")
    if not optimizer.arguments:
        errors.append("OPTIMIZER.arguments must provide keyword args.")

    lr_sched = cfg.SCHEDULER.learning_rate
    if lr_sched.enabled:
        if not lr_sched.name:
            errors.append("SCHEDULER.learning_rate.name must be specified when enabled.")
        if not isinstance(lr_sched.arguments, dict):
            errors.append("SCHEDULER.learning_rate.arguments must be a mapping.")

    if errors:
        error_text = "\n - ".join(errors)
        raise ValueError(f"Configuration validation failed:\n - {error_text}")


def load_config_with_schema(path: str) -> Tuple[AppConfig, Dict[str, Any], ConfigUsageTracker]:
    user_cfg = OmegaConf.load(path)
    structured_default = OmegaConf.structured(AppConfig)
    merged_cfg = OmegaConf.merge(structured_default, user_cfg)
    cfg_obj: AppConfig = OmegaConf.to_object(merged_cfg)
    validate_app_config(cfg_obj)
    tracker = ConfigUsageTracker()
    attach_tracker(cfg_obj, tracker)
    return cfg_obj, tracker


def app_config_to_dict(cfg: AppConfig) -> Dict[str, Any]:
    return asdict(cfg)
