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
import inspect
import yaml

import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc

from argparse import ArgumentParser
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.table import Table

from model_interface import ModelInterface
from data_interface import DataInterface
from utils.logging import get_resume_info


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
        """
        Format metrics into a Rich table with multiple rows.
        """
        table = Table(show_header=False, box=None, expand=True)

        keys = list(metrics.keys())
        for i in range(0, len(keys), self.metrics_per_row):
            chunk_keys = keys[i : i + self.metrics_per_row]
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
        """
        Override RichProgressBar.render to display metrics as multi-row table.
        """
        renderables = super().render(*args, **kwargs)
        metrics = self.get_metrics(self._trainer, self._trainer.lightning_module)

        if renderables and metrics:
            renderables[-1] = self._render_metrics_table(metrics)

        return renderables


# For all built-in callback functions, see: https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
# The following callback functions are commonly used and ready to load based on user setting.
def load_callbacks(config):
    callbacks = []

    callbacks.append(MultiRowRichProgressBar(refresh_rate=1, leave=False, metrics_per_row=5))

    # Monitor a metric and stop training when it stops improving
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss_epoch',
        mode='min',
        patience=10,
        min_delta=0.001
    ))

    # Save the model periodically by monitoring a quantity
    if config['enable_checkpointing']:
        # Best & Last checkpoint
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=1,
            monitor='val_loss_epoch',
            mode='min',
            filename='best-{epoch:03d}-{val_loss_epoch:.5f}',
            save_top_k=1,
            save_last=True,
        ))

    # Monitor learning rate decay
    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'
    ))

    # Generates a summary of all layers in a LightningModule based on max_depth.
    """
    callbacks.append(plc.ModelSummary(
        max_depth=1
    ))
    """

    # Change gradient accumulation factor according to scheduling.
    # Only consider using this when batch_size does not fit into current hardware environment.
    """
    callbacks.append(plc.GradientAccumulationScheduler(
        # From epoch 5, it starts accumulating every 4 batches. Here we have 4 instead of 5 because epoch (key) should be zero-indexed.
        scheduling={4: 4}
    ))
    """

    return callbacks


def main(config):
    pl.seed_everything(config['seed'])

    resume_info = get_resume_info(config)
    mode       = resume_info['mode']
    run_name   = resume_info['run_name']    # logger 'name'
    version    = resume_info['version']     # logger 'version' (None → new sibling)
    ckpt_path  = resume_info['ckpt_path']

    # --- Logger placement
    # Full-resume → log into the SAME run/version (no nested version folders)
    # Warm-start/Scratch → SAME run name but NEW version (version=None)
    if mode == 'resume':
        logger = TensorBoardLogger(save_dir='.', name=run_name, version=version)
        config['log_dir_full'] = os.path.join(run_name, version)
    else:
        logger = TensorBoardLogger(save_dir='.', name=run_name, version=None)  # allocate version_0/1/2...
        # We don't know the resolved version name yet; Lightning decides. Store parent for convenience.
        config['log_dir_full'] = run_name

    weights_only = bool(config.get('weights_only', False))
    strict_sd    = bool(config.get('strict_state_dict', True))
    map_loc      = config.get('map_location', None)  # e.g., "cpu" or torch.device("cpu")
    if 'weights_only' in config:
        config.pop('weights_only')
    if 'map_location' in config:
        config.pop('map_location')
    if 'strict_state_dict' in config:
        config.pop('strict_state_dict')

    if (mode == 'warmstart') and ckpt_path:
        # Load only weights; start from epoch 0 and new global_step
        model_module = ModelInterface.load_from_checkpoint(
            ckpt_path, strict=strict_sd, map_location=map_loc, **config
        )
        ckpt_for_trainer_fit = None
    else:
        # Fresh module; if mode == 'resume', Trainer will restore full state via ckpt_path
        model_module = ModelInterface(**config)
        ckpt_for_trainer_fit = ckpt_path if (mode == 'resume') else None

    # --- Data & model
    data_module = DataInterface(**config)

    # --- Trainer args
    config['logger'] = logger
    config['callbacks'] = load_callbacks(config=config)

    signature = inspect.signature(Trainer.__init__)
    filtered_trainer_keywords = {k: config[k] for k in signature.parameters.keys() if k in config}

    trainer = Trainer(**filtered_trainer_keywords)

    # Full resume → pass ckpt_path so optimizer/scheduler/global_step are restored.
    # Warm-start/scratch → pass None so training begins fresh (but with loaded weights for warm-start).
    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=ckpt_for_trainer_fit)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config_path', default=os.path.join(os.getcwd(), 'config', 'config.yaml'),
                        type=str, required=False, help='Path of config file')
    parser.add_argument('--resume_from_last_checkpoint', action='store_true',
                        help='Auto-find the latest run/version and resume from the newest checkpoint inside its "checkpoints" dir. ' \
                        'This implies full-resume of all states, which means weights_only will be ignored')
    parser.add_argument('--load_manual_checkpoint', default=None, type=str, required=False,
                        help='Path to a .ckpt file to load. Combine with --weights_only to warm-start.')
    parser.add_argument('--weights_only', action='store_true',
                        help='Warm-start weights only (do not restore optimizer/scheduler/global_step).')
    parser.add_argument('--strict_state_dict', action='store_true',
                        help='If set (default True), require exact state_dict match when warm-starting.')
    parser.add_argument('--map_location', default=None, type=str, required=False,
                        help='Optional map_location for warm-start loading, e.g., "cpu", "cuda".')

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')

    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    # CLI overrides
    config_dict['load_manual_checkpoint'] = args.load_manual_checkpoint
    config_dict['resume_from_last_checkpoint'] = args.resume_from_last_checkpoint
    config_dict['weights_only'] = getattr(args, 'weights_only', False)
    config_dict['strict_state_dict'] = getattr(args, 'strict_state_dict', True)
    config_dict['map_location'] = getattr(args, 'map_location', None)

    # normalize keys to lowercase
    config_dict = dict((k.lower(), v) for k, v in config_dict.items())

    main(config_dict)