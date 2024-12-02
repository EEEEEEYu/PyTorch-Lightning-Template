from dataclasses import dataclass

from configs.config_tracker import TrackedConfigMixin

@dataclass
class LoggerConfig(TrackedConfigMixin):
    log_dir_root: str = "lightning_logs"
    experiment_name: str = "experiment"
