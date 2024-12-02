import os
import re
import datetime
from typing import Any, Dict
from configs.config_schema import AppConfig

_VERSION_RE = re.compile(r"^version_(\d+)$")


def _pick_latest_ckpt(ckpt_dir: str):
    """Pick the newest-looking checkpoint file under <version_dir>/checkpoints."""
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".ckpt"):
            continue
        if fn == "last.ckpt" or fn.startswith("latest") or "epoch=" in fn:
            full = os.path.join(ckpt_dir, fn)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            candidates.append((mtime, full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_latest_run_dir(root_dir: str):
    """Return the most recently modified run directory under root_dir, or None."""
    if not os.path.isdir(root_dir):
        return None
    entries = []
    for d in os.listdir(root_dir):
        full = os.path.join(root_dir, d)
        if os.path.isdir(full):
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            entries.append((mtime, full))
    if not entries:
        return None
    entries.sort(reverse=True)
    return entries[0][1]


def _find_latest_version_dir(run_dir: str):
    """Return the version_* dir with the highest numeric suffix (or most recent mtime)."""
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for d in os.listdir(run_dir):
        m = _VERSION_RE.match(d)
        if not m:
            continue
        full = os.path.join(run_dir, d)
        if not os.path.isdir(full):
            continue
        idx = int(m.group(1))
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0
        candidates.append((idx, mtime, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _parse_run_and_version_from_ckpt(ckpt_path: str):
    """
    Given .../<run_name>/<version_x>/checkpoints/<file>.ckpt,
    return (run_name_rel, version_dir_name, version_dir_abs, run_dir_abs).
    If not matchable, return fallbacks.
    """
    checkpoints_dir = os.path.dirname(ckpt_path)
    version_dir = os.path.dirname(checkpoints_dir)
    run_dir = os.path.dirname(version_dir)
    version_name = os.path.basename(version_dir)
    run_name_rel = os.path.relpath(run_dir, start='.')
    return run_name_rel, version_name, version_dir, run_dir


def get_resume_info(config: AppConfig, runtime: Dict[str, Any]):
    """
    Decide how to start:
      - full-resume: restore full trainer state from a checkpoint
      - warm-start:  load weights only, start new version under same run
      - scratch:     new run + new version

    Returns a dict with keys:
      mode: 'resume' | 'warmstart' | 'scratch'
      run_name: str (relative path used as TensorBoardLogger.name)
      version:  str or None (e.g., 'version_0' on resume, None otherwise)
      ckpt_path: str or None
    """
    checkpoint_cfg = config.CHECKPOINT
    logger_cfg = config.LOGGER

    enable_ckpt = checkpoint_cfg.enabled
    log_dir_root = logger_cfg.log_dir_root
    experiment_name = logger_cfg.experiment_name

    if not enable_ckpt:
        run_name = os.path.join(
            log_dir_root,
            f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{experiment_name}"
        )
        return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

    load_manual_checkpoint = runtime.get('load_manual_checkpoint')
    resume_from_last_checkpoint = runtime.get('resume_from_last_checkpoint')
    weights_only = bool(runtime.get('weights_only', False))

    if load_manual_checkpoint:
        if not os.path.isfile(load_manual_checkpoint):
            raise FileNotFoundError(f"Manual checkpoint not found: {load_manual_checkpoint}")
        run_name, version_name, _, _ = _parse_run_and_version_from_ckpt(load_manual_checkpoint)

        if weights_only:
            return dict(mode='warmstart', run_name=run_name, version=None, ckpt_path=load_manual_checkpoint)
        return dict(mode='resume', run_name=run_name, version=version_name, ckpt_path=load_manual_checkpoint)

    if resume_from_last_checkpoint:
        latest_run_dir = _find_latest_run_dir(log_dir_root)
        if latest_run_dir is None:
            print(f"Warning: resume_from_last_checkpoint=True but no runs under {log_dir_root}. Starting new training.")
            run_name = os.path.join(
                log_dir_root,
                f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{experiment_name}"
            )
            return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

        version_dir = _find_latest_version_dir(latest_run_dir)
        if version_dir is None:
            print(f"Warning: No version_* directories under {latest_run_dir}. Starting new training.")
            run_name = os.path.join(
                log_dir_root,
                f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{experiment_name}"
            )
            return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

        ckpt_path = _pick_latest_ckpt(os.path.join(version_dir, 'checkpoints'))
        if ckpt_path is None:
            print(f"Warning: No checkpoint files in {os.path.join(version_dir, 'checkpoints')}. Starting new training.")
            run_name = os.path.join(
                log_dir_root,
                f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{experiment_name}"
            )
            return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

        run_name_rel = os.path.relpath(os.path.dirname(version_dir), start='.')
        version_name = os.path.basename(version_dir)

        return dict(mode='resume', run_name=run_name_rel, version=version_name, ckpt_path=ckpt_path)

    run_name = os.path.join(
        log_dir_root,
        f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{experiment_name}"
    )
    return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)
