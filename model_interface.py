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

import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as pl

from utils.metrics.classification import top1_accuracy, top5_accuracy
from loss.loss_funcs import cross_entropy_loss
from typing import Callable, Dict, Tuple


class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()
        self.train_epoch_output = []
        self.val_epoch_output = []
        self.test_epoch_output = []

    def forward(self, x):
        return self.model(x)
    
    # For all these hook functions like on_XXX_<epoch|batch>_<end|start>(), 
    # check document: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
    # Epoch level training logging
    def on_train_epoch_end(self):
        train_top1_acc = top1_accuracy(self.train_epoch_output)
        train_top5_acc = top5_accuracy(self.train_epoch_output)

        # Metrics logging. If on_step=True, lightning will log this metric with a '_step' suffix.
        # If on_epoch=True, lightning will by default use mean reduction to aggregate step metrics and
        # log this metric with a '_epoch' suffix
        self.log('train_top1_accuracy', train_top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_top5_accuracy', train_top5_acc, on_step=False, on_epoch=True, prog_bar=True)

        del self.train_epoch_output
        self.train_epoch_output = []

    # Epoch level validation logging
    def on_validation_epoch_end(self):
        val_top1_acc = top1_accuracy(self.val_epoch_output)
        val_top5_acc = top5_accuracy(self.val_epoch_output)

        # Metrics logging. If on_step=True, lightning will log this metric with a '_step' suffix.
        # If on_epoch=True, lightning will by default use mean reduction to aggregate step metrics and
        # log this metric with a '_epoch' suffix
        self.log('val_top1_accuracy', val_top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_top5_accuracy', val_top5_acc, on_step=False, on_epoch=True, prog_bar=True)

        del self.val_epoch_output
        self.val_epoch_output = []

    # Epoch level testing logging
    def on_test_epoch_end(self):
        test_top1_acc = top1_accuracy(self.test_epoch_output)
        test_top5_acc = top5_accuracy(self.test_epoch_output)

        # Metrics logging. If on_step=True, lightning will log this metric with a '_step' suffix.
        # If on_epoch=True, lightning will by default use mean reduction to aggregate step metrics and
        # log this metric with a '_epoch' suffix
        self.log('test_top1_accuracy', test_top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_top5_accuracy', test_top5_acc, on_step=False, on_epoch=True, prog_bar=True)

        del self.test_epoch_output
        self.test_epoch_output = []

    # Caution: self.model.train() is invoked
    def training_step(self, batch, batch_idx):
        train_input, train_labels = batch
        train_out = self(train_input)
        train_loss = self.loss_function(train_out, train_labels, 'train')

        train_step_output = {
            'loss': train_loss,
            'pred': train_out,
            'ground_truth': train_labels
        }

        self.train_epoch_output.append(train_step_output)

        return train_step_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        val_input, val_labels = batch
        val_out = self(val_input)
        val_loss = self.loss_function(val_out, val_labels, 'val')

        val_step_output = {
            'loss': val_loss,
            'pred': val_out,
            'ground_truth': val_labels
        }

        self.val_epoch_output.append(val_step_output)

        return val_step_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        test_input, test_labels = batch
        test_out = self(test_input)
        test_loss = self.loss_function(test_out, test_labels, 'test')

        test_step_output = {
            'loss': test_loss,
            'pred': test_out,
            'ground_truth': test_labels
        }

        self.test_epoch_output.append(test_step_output)

        return test_step_output

    # When there are multiple optimizers, modify this function to fit in your needs
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        # No learning rate scheduler, just return the optimizer
        if self.hparams.lr_scheduler is None:
            return [optimizer]

        # Return tuple of optimizer and learning rate scheduler
        if self.hparams.lr_scheduler == 'step':
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_epochs,
                gamma=self.hparams.lr_decay_rate
            )
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_decay_epochs,
                eta_min=self.hparams.lr_decay_min_lr
            )
        else:
            raise ValueError('Invalid lr_scheduler type!')
        return [optimizer], [scheduler]

    def __configure_loss(self):
        def loss_func(preds, labels, stage):
            # Calculate and log each component individually
            CE_loss = 1.0 * cross_entropy_loss(pred=preds, gt=labels)
            self.log(f'{stage}_CE_loss', CE_loss, on_step=True, on_epoch=True, prog_bar=True)

            # Log the final compound loss
            final_loss = CE_loss
            self.log(f'{stage}_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True)

            return final_loss

        return loss_func

    def __load_model(self):
        name = self.hparams.model_class_name
        # Attempt to import the `CamelCase` class name from the `snake_case.py` module. The module should be placed
        # within the same folder as model_interface.py. Always name your model file name as `snake_case.py` and
        # model class name as corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model_class = getattr(importlib.import_module('model.' + name, package=__package__), camel_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        model = self.__instantiate(model_class)
        if self.hparams.use_compile:
            torch.compile(model)
        return model

    def __instantiate(self, model_class, **other_args):
        # Instantiate a model using the imported class name and parameters from self.hparams dictionary.
        # You can also input any args to overwrite the corresponding value in self.hparams.
        target_args = inspect.getfullargspec(model_class.__init__).args[1:]
        this_args = self.hparams.keys()
        merged_args = {}
        # Only assign arguments that are required in the user-defined torch.nn.Module subclass by their name.
        # You need to define the required arguments in main function.
        for arg in target_args:
            if arg in this_args:
                merged_args[arg] = getattr(self.hparams, arg)

        merged_args.update(other_args)
        return model_class(**merged_args)
