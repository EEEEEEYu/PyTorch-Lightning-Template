from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Optional, Set, Any, ClassVar


class ConfigUsageTracker:
    """Tracks which config paths exist and which are accessed at runtime."""

    def __init__(self):
        self._declared: Set[str] = set()
        self._used: Set[str] = set()

    def register(self, path: str):
        if path:
            self._declared.add(path)

    def mark_used(self, path: str):
        if path:
            self._used.add(path)

    def unused_paths(self):
        return sorted(self._declared - self._used)

    def report(self):
        unused = self.unused_paths()
        if unused:
            print("Unused config entries detected:")
            for path in unused:
                print(f" - {path}")
        else:
            print("All tracked config entries were accessed during runtime.")


class TrackedConfigMixin:
    _config_tracker: ClassVar[Optional[ConfigUsageTracker]] = None
    _config_path: ClassVar[str] = ""
    _config_frozen: ClassVar[bool] = False

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "_config_tracker",
            "_config_path",
            "_config_frozen",
            "_set_tracker",
            "__class__",
            "__dict__",
            "__module__",
        } or name.startswith("__"):
            return object.__getattribute__(self, name)
        value = object.__getattribute__(self, name)
        tracker = object.__getattribute__(self, "_config_tracker")
        if tracker is not None and not name.startswith("_config_"):
            path = object.__getattribute__(self, "_config_path")
            child_path = f"{path}.{name}" if path else name
            tracker.mark_used(child_path)
        return value

    def _set_tracker(self, tracker: ConfigUsageTracker, path: str):
        object.__setattr__(self, "_config_tracker", tracker)
        object.__setattr__(self, "_config_path", path)
        object.__setattr__(self, "_config_frozen", True)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_config_"):
            object.__setattr__(self, name, value)
            return
        frozen = object.__getattribute__(self, "_config_frozen")
        if frozen:
            raise AttributeError(
                f"Configuration is read-only after initialization. Attempted to set '{name}' on {self.__class__.__name__}."
            )
        object.__setattr__(self, name, value)


def attach_tracker(obj: Any, tracker: ConfigUsageTracker, path: str = ""):
    if isinstance(obj, TrackedConfigMixin):
        obj._set_tracker(tracker, path)
        if path:
            tracker.register(path)
        for field in fields(obj):
            name = field.name
            if name.startswith("_config_"):
                continue
            child_path = f"{path}.{name}" if path else name
            tracker.register(child_path)
            value = object.__getattribute__(obj, name)
            attach_tracker(value, tracker, child_path)
    elif is_dataclass(obj):
        if path:
            tracker.register(path)
        for field in fields(obj):
            child_path = f"{path}.{field.name}" if path else field.name
            tracker.register(child_path)
            value = getattr(obj, field.name)
            attach_tracker(value, tracker, child_path)
