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
import datetime
import inspect
import pathlib
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


def get_checkpoint_path(config):
    # Check if resuming from a manual checkpoint or last checkpoint
    load_manual_checkpoint = config.get('load_manual_checkpoint', None)
    resume_from_last_checkpoint = config.get('resume_from_last_checkpoint', None)
    checkpoint_directory = None
    checkpoint_file_path = None

    if load_manual_checkpoint:
        # After truncating the path, PL will automatically append a new version under the parent folder(which has format: {timestamp}-{experiment name})
        checkpoint_file_path = load_manual_checkpoint
        truncated_path = load_manual_checkpoint
        for i in range(2):
            truncated_path = truncated_path[:truncated_path.rfind(os.path.sep)]
        # Use the same logging directory as the checkpoint
        checkpoint_directory = os.path.dirname(truncated_path)
    # Find the latest folder with latest version
    elif resume_from_last_checkpoint:
        # Find the latest log folder
        current_path = config['log_dir_root']
        logs_with_datetime = os.listdir(config['log_dir_root'])
        logs_with_datetime.sort(reverse=True)
        if len(logs_with_datetime) == 0 or len(os.listdir(os.path.join(current_path, logs_with_datetime[0]))) == 0:
            print(f"Warning: resume_from_last_checkpoint was set to True but no checkpoint found at: {current_path}. Launch a new training...")
            return None, None
        
        # Find the latest version folder
        checkpoint_directory = os.path.join(current_path, logs_with_datetime[0])
        checkpoint_file_path = os.path.join(checkpoint_directory, 'checkpoints')
        if not any(s.startswith('latest') and s.endswith('.ckpt') for s in os.listdir(checkpoint_file_path)):
            print(f"Warning: resume_from_last_checkpoint was set to True but no checkpoint file found at: {checkpoint_file_path}. Launch a new training...")
            return None, None

        # Find the latest checkpoint file with largest epoch
        ckpt_files = list(filter(lambda s: s.startswith('latest') and s.endswith('.ckpt'), os.listdir(checkpoint_file_path)))
        ckpt_files.sort(reverse=True)
        checkpoint_file_path = os.path.join(checkpoint_file_path, ckpt_files[0])

    return checkpoint_directory, checkpoint_file_path


def main(config):
    # Set random seed
    pl.seed_everything(config['seed'])        

    checkpoint_directory, checkpoint_file_path = get_checkpoint_path(config=config) if config['enable_checkpointing'] else (None, None)

    # Caution: the final checkpoint directory depends on the logger path
    # When the loaded checkpoint reached max_epoch, the training will stop immediately
    if checkpoint_directory is not None:
        logger = TensorBoardLogger(save_dir='.', name=checkpoint_directory, version=None)
        log_dir_full = pathlib.Path(checkpoint_directory).parent
        config['log_dir_full'] = str(log_dir_full)
    else:
        # Create a new logging directory
        print("Training from scratch...")
        log_dir_name_with_time = os.path.join(config['log_dir_root'], datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S"))
        log_dir_full = f"{log_dir_name_with_time}-{config['experiment_name']}"
        config['log_dir_full'] = log_dir_full
        logger = TensorBoardLogger(save_dir='.', name=log_dir_full, version=None)

    # Instantiate model and data module
    data_module = DataInterface(**config)
    model_module = ModelInterface(**config) 
    
    # Add logger
    config['logger'] = logger

    # Load callback functions for Trainer
    config['callbacks'] = load_callbacks(config=config)

    # Add resume_from_checkpoint to the trainer initialization
    signature = inspect.signature(Trainer.__init__)
    filtered_trainer_keywords = {}
    for arg in list(signature.parameters.keys()):
        if arg in config:
            filtered_trainer_keywords[arg] = config[arg]

    # Instantiate the Trainer object
    trainer = Trainer(**filtered_trainer_keywords)

    # Launch the training
    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=checkpoint_file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', default=os.path.join(os.getcwd(), 'config', 'config.yaml'), type=str, required=False,
                        help='Path of config file')
    parser.add_argument('--resume_from_last_checkpoint', action='store_true',
                    help='Automatically find the log folder with latest timestamp and latest version, and load `latest-...`.ckpt model')
    parser.add_argument('--load_manual_checkpoint', default=None, type=str, required=False,
                    help='Manually designate the path to a checkpoint file(.ckpt) to resume training from.')

    # Parse arguments(set attributes for sys.args using above arguments)
    args = parser.parse_args()

    # Validate config file and dataset directory
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')

    # Load .yaml config file as python dict
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    # Convert config dict keys to lowercase
    config_dict['load_manual_checkpoint'] = args.load_manual_checkpoint
    config_dict['resume_from_last_checkpoint'] = args.resume_from_last_checkpoint
    config_dict = dict([(k.lower(), v) for k, v in config_dict.items()])

    # Activate main function
    main(config_dict)
