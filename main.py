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

import yaml
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import inspect

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ModelInterface
from data import DataInterface


# For all built-in callback functions, see: https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
# The following callback functions are commonly used and ready to load based on user setting.
def load_callbacks():
    callbacks = []
    # Monitor a metric and stop training when it stops improving
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    # Save the model periodically by monitoring a quantity
    callbacks.append(plc.ModelCheckpoint(
        every_n_epochs=1,
        monitor='val_acc',
        mode='max',
        filename='best-{epoch:03d}-{val_acc:.3f}',
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
    # Set random seed
    pl.seed_everything(config['seed'])

    # Instantiate model and data module. Pass the arguments by unpacking the config dict.
    data_module = DataInterface(**config)
    model_module = ModelInterface(**config)

    # Add the tensorboard logger
    log_dir_name_with_time = os.path.join(config['log_dir_name'], datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S"))
    logger = TensorBoardLogger(save_dir=config['log_dir'], name=f"{log_dir_name_with_time}-{config['experiment_name']}")
    config['logger'] = logger

    # Load callback functions for Trainer
    config['callbacks'] = load_callbacks()

    # Instantiate the Trainer object
    signature = inspect.signature(Trainer.__init__)
    filtered_trainer_keywords = {}
    for arg in list(signature.parameters.keys()):
        if arg in config:
            filtered_trainer_keywords[arg] = config[arg]
    trainer = Trainer(**filtered_trainer_keywords)

    # Launch the training
    trainer.fit(model=model_module, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', default=os.path.join(os.getcwd(), 'config', 'config.yaml'), type=str, required=False,
                        help='Path of config file')
    parser.add_argument('--dataset_dir', default=os.path.join(os.getcwd(), 'dataset'), type=str, required=False,
                        help='Directory of dataset')
    # Parse arguments(set attributes for sys.args using above arguments)
    args = parser.parse_args()

    # Validate config file and dataset directory
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')

    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f'No dataset directory found at {args.dataset_dir}')

    # Load .yaml config file as python dict
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    # Convert config dict keys to lowercase
    config_dict = dict([(k.lower(), v) for k, v in config_dict.items()])

    # Activate main function
    main(config_dict)
