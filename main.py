# Copyright 2024 Haowen Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
from argparse import ArgumentParser
from typing import Any, Dict

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.table import Table

from model_interface import ModelInterface
from data_interface import DataInterface
from utils.logging import get_resume_info
from configs.config_schema import load_config_with_schema, AppConfig
from configs.config_tracker import ConfigUsageTracker


class MultiRowRichProgressBar(RichProgressBar):
    def __init__(self, metrics_per_row: int = 4, refresh_rate: int = 1, leave: bool = False):
        super().__init__(theme=RichProgressBarTheme())
        self.metrics_per_row = metrics_per_row

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if "v_num" in metrics:
            del metrics["v_num"]
        return metrics

    def _render_metrics_table(self, metrics: dict) -> Table:
        table = Table(show_header=False, box=None, expand=True)

        keys = list(metrics.keys())
        for i in range(0, len(keys), self.metrics_per_row):
            chunk_keys = keys[i: i + self.metrics_per_row]
            row = []
            for k in chunk_keys:
                v = metrics[k]
                if isinstance(v, (int, float)):
                    row.append(f"{k}: {v:.5f}")
                else:
                    row.append(f"{k}: {v}")
            table.add_row(*row)

        return table

    def render(self, *args, **kwargs):
        renderables = super().render(*args, **kwargs)
        metrics = self.get_metrics(self._trainer, self._trainer.lightning_module)

        if renderables and metrics:
            renderables[-1] = self._render_metrics_table(metrics)

        return renderables


def load_callbacks(cfg: AppConfig):
    callbacks = []

    callbacks.append(MultiRowRichProgressBar(refresh_rate=1, leave=False, metrics_per_row=5))
    callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    early_stopping_cfg = cfg.SCHEDULER.early_stopping
    if early_stopping_cfg.enabled:
        callbacks.append(plc.EarlyStopping(
            monitor=early_stopping_cfg.monitor,
            mode=early_stopping_cfg.mode,
            patience=early_stopping_cfg.patience,
            min_delta=early_stopping_cfg.min_delta
        ))

    checkpoint_cfg = cfg.CHECKPOINT
    if checkpoint_cfg.enabled:
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=checkpoint_cfg.every_n_epochs,
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            filename=checkpoint_cfg.filename,
            save_top_k=checkpoint_cfg.save_top_k,
            save_last=checkpoint_cfg.save_last,
        ))

    gradient_accumulation_scheduler_cfg = cfg.OPTIMIZER.gradient_accumulation
    if gradient_accumulation_scheduler_cfg.enabled:
        callbacks.append(plc.GradientAccumulationScheduler(
            scheduling=gradient_accumulation_scheduler_cfg.scheduling
        ))

    stochastic_weight_averaging_cfg = cfg.OPTIMIZER.stochastic_weight_averaging
    if stochastic_weight_averaging_cfg.enabled:
        callbacks.append(plc.StochasticWeightAveraging(
            swa_lrs=stochastic_weight_averaging_cfg.swa_lrs
        ))


    return callbacks


def _build_trainer_kwargs(cfg: AppConfig, logger: TensorBoardLogger, callbacks: list) -> Dict[str, Any]:
    training_cfg = cfg.TRAINING
    distributed_cfg = cfg.DISTRIBUTED
    optimizer_cfg = cfg.OPTIMIZER

    trainer_kwargs = {
        "max_epochs": training_cfg.max_epochs,
        "deterministic": training_cfg.deterministic,
        "inference_mode": training_cfg.inference_mode,
        "logger": logger,
        "callbacks": callbacks,
        "accelerator": distributed_cfg.accelerator,
        "devices": distributed_cfg.devices,
        "num_nodes": distributed_cfg.num_nodes,
        "strategy": distributed_cfg.strategy,
    }

    gradient_clip_cfg = optimizer_cfg.gradient_clip
    if gradient_clip_cfg.enabled:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_cfg.gradient_clip_val
        trainer_kwargs["gradient_clip_algorithm"] = gradient_clip_cfg.gradient_clip_algorithm

    return {k: v for k, v in trainer_kwargs.items() if v is not None}


def main(cfg: AppConfig, tracker: ConfigUsageTracker, runtime: Dict[str, Any]):
    runtime = copy.deepcopy(runtime)

    seed = cfg.TRAINING.seed
    pl.seed_everything(seed)

    resume_info = get_resume_info(cfg, runtime)
    mode = resume_info['mode']
    run_name = resume_info['run_name']
    version = resume_info['version']
    ckpt_path = resume_info['ckpt_path']

    logger = TensorBoardLogger(save_dir='.', name=run_name, version=version if mode == 'resume' else None)

    model_interface_kwargs = {
        "model_cfg": cfg.MODEL,
        "optimizer_cfg": cfg.OPTIMIZER,
        "scheduler_cfg": cfg.SCHEDULER,
        "training_cfg": cfg.TRAINING,
        "data_cfg": cfg.DATA,
    }

    data_interface_kwargs = {
        "data_cfg": cfg.DATA
    }

    if mode == 'warmstart' and ckpt_path:
        model_module = ModelInterface.load_from_checkpoint(
            ckpt_path,
            strict=bool(runtime.get('strict_state_dict', True)),
            map_location=runtime.get('map_location', None),
            **model_interface_kwargs,
        )
        ckpt_for_trainer_fit = None
    else:
        model_module = ModelInterface(**model_interface_kwargs)
        ckpt_for_trainer_fit = ckpt_path if mode == 'resume' else None

    data_module = DataInterface(**data_interface_kwargs)

    callbacks = load_callbacks(cfg)
    trainer_kwargs = _build_trainer_kwargs(cfg, logger, callbacks)
    trainer = Trainer(**trainer_kwargs)

    try:
        trainer.fit(model=model_module, datamodule=data_module, ckpt_path=ckpt_for_trainer_fit)
    finally:
        tracker.report()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', default=os.path.join(os.getcwd(), 'config', 'config.yaml'),
                        type=str, required=False, help='Path of config file')
    parser.add_argument('--resume_from_last_checkpoint', action='store_true',
                        help='Auto-find the latest run/version and resume from the newest checkpoint inside its "checkpoints" dir. '
                             'This implies full-resume of all states, which means weights_only will be ignored')
    parser.add_argument('--load_manual_checkpoint', default=None, type=str, required=False,
                        help='Path to a .ckpt file to load. Combine with --weights_only to warm-start.')
    parser.add_argument('--weights_only', action='store_true',
                        help='Warm-start weights only (do not restore optimizer/scheduler/global_step).')
    parser.add_argument('--strict_state_dict', action='store_true',
                        help='If set, require exact state_dict match when warm-starting. Enabled by default.')
    parser.add_argument('--no_strict_state_dict', dest='strict_state_dict', action='store_false',
                        help='Disable strict state_dict matching when warm-starting.')
    parser.add_argument('--map_location', default=None, type=str, required=False,
                        help='Optional map_location for warm-start loading, e.g., "cpu", "cuda".')
    parser.set_defaults(strict_state_dict=True)

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')

    cfg_obj, cfg_tracker = load_config_with_schema(args.config_path)

    runtime_cfg = {
        'load_manual_checkpoint': args.load_manual_checkpoint,
        'resume_from_last_checkpoint': args.resume_from_last_checkpoint,
        'weights_only': getattr(args, 'weights_only', False),
        'strict_state_dict': getattr(args, 'strict_state_dict', True),
        'map_location': getattr(args, 'map_location', None),
    }

    main(cfg_obj, cfg_tracker, runtime_cfg)
