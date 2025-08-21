import os
import re
import datetime
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^version_(\d+)$")

def _pick_latest_ckpt(ckpt_dir: str):
    """Pick the newest-looking checkpoint file under <version_dir>/checkpoints."""
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".ckpt"):
            continue
        # Common names we support: last.ckpt, latest-*.ckpt, epoch=*.ckpt
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
    # Prefer highest version index; break ties by mtime
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

def _parse_run_and_version_from_ckpt(ckpt_path: str):
    """
    Given .../<run_name>/<version_x>/checkpoints/<file>.ckpt,
    return (run_name_rel, version_dir_name, version_dir_abs, run_dir_abs).
    If not matchable, return fallbacks.
    """
    checkpoints_dir = os.path.dirname(ckpt_path)
    version_dir = os.path.dirname(checkpoints_dir)  # .../<run_name>/<version_x>
    run_dir = os.path.dirname(version_dir)          # .../<run_name>
    version_name = os.path.basename(version_dir)    # version_x
    run_name_rel = os.path.relpath(run_dir, start='.')
    return run_name_rel, version_name, version_dir, run_dir

# --------------------------------------------------------------------------------------
# Checkpoint discovery
# --------------------------------------------------------------------------------------

def get_resume_info(config):
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
    enable_ckpt = config.get('enable_checkpointing', True)
    if not enable_ckpt:
        # Training from scratch, new run
        run_name = os.path.join(
            config['log_dir_root'],
            f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{config['experiment_name']}"
        )
        return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

    load_manual_checkpoint = config.get('load_manual_checkpoint', None)
    resume_from_last_checkpoint = config.get('resume_from_last_checkpoint', None)
    weights_only = bool(config.get('weights_only', False))

    if load_manual_checkpoint:
        # Use the run/version parsed from the manual checkpoint
        if not os.path.isfile(load_manual_checkpoint):
            raise FileNotFoundError(f"Manual checkpoint not found: {load_manual_checkpoint}")
        run_name, version_name, _, run_dir_abs = _parse_run_and_version_from_ckpt(load_manual_checkpoint)

        if weights_only:
            # Warm-start into a NEW sibling version under the SAME run
            return dict(mode='warmstart', run_name=run_name, version=None, ckpt_path=load_manual_checkpoint)
        else:
            # Full resume into the SAME run/version (continuous TB history)
            return dict(mode='resume', run_name=run_name, version=version_name, ckpt_path=load_manual_checkpoint)

    if resume_from_last_checkpoint:
        # Find latest run under log_dir_root
        latest_run_dir = _find_latest_run_dir(config['log_dir_root'])
        if latest_run_dir is None:
            print(f"Warning: resume_from_last_checkpoint=True but no runs under {config['log_dir_root']}. Starting new training.")
            run_name = os.path.join(
                config['log_dir_root'],
                f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{config['experiment_name']}"
            )
            return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

        # Find the latest version dir inside that run
        version_dir = _find_latest_version_dir(latest_run_dir)
        if version_dir is None:
            print(f"Warning: No version_* directories under {latest_run_dir}. Starting new training.")
            run_name = os.path.join(
                config['log_dir_root'],
                f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{config['experiment_name']}"
            )
            return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

        ckpt_path = _pick_latest_ckpt(os.path.join(version_dir, 'checkpoints'))
        if ckpt_path is None:
            print(f"Warning: No checkpoint files in {os.path.join(version_dir, 'checkpoints')}. Starting new training.")
            run_name = os.path.join(
                config['log_dir_root'],
                f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{config['experiment_name']}"
            )
            return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

        run_name_rel = os.path.relpath(os.path.dirname(version_dir), start='.')
        version_name = os.path.basename(version_dir)

        # resume_from_last_checkpoint implies full resume
        return dict(mode='resume', run_name=run_name_rel, version=version_name, ckpt_path=ckpt_path)

    # Default: scratch new run
    run_name = os.path.join(
        config['log_dir_root'],
        f"{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}-{config['experiment_name']}"
    )
    return dict(mode='scratch', run_name=run_name, version=None, ckpt_path=None)

