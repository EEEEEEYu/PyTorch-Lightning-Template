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
import pytorch_lightning as pl

from loss.loss_funcs import cross_entropy_loss
from typing import Callable, Dict, Tuple


class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()
        self.train_epoch_correct = 0
        self.train_epoch_total = 0
        self.val_epoch_correct = 0
        self.val_epoch_total = 0

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_end(self):
        self.log('train_accuracy', float(self.train_epoch_correct) / float(self.train_epoch_total), on_step=False, on_epoch=True, prog_bar=True)
        self.train_epoch_correct = 0
        self.train_epoch_total = 0

    def on_validation_epoch_end(self):
        self.log('val_accuracy', float(self.val_epoch_correct) / float(self.val_epoch_total), on_step=False, on_epoch=True, prog_bar=True)
        self.val_epoch_correct = 0
        self.val_epoch_correct = 0

    # Caution: self.model.train() is invoked
    def training_step(self, batch, batch_idx):
        train_input, train_labels = batch
        B = train_labels.shape[0]
        train_out = self(train_input)
        train_loss = self.loss_function(train_out, train_labels, 'train')

        label_digit = train_labels.argmax(axis=1)
        out_digit = train_out.argmax(axis=1)
        correct_num = torch.sum(label_digit == out_digit).cpu().item()

        self.train_epoch_correct += correct_num
        self.train_epoch_total += B
        self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', float(correct_num) / float(B), on_step=True, on_epoch=False, prog_bar=True)

        return train_loss

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        val_input, val_labels = batch
        B = val_labels.shape[0]
        val_out = self(val_input)
        val_loss = self.loss_function(val_out, val_labels, 'validation')

        label_digit = val_labels.argmax(axis=1)
        out_digit = val_out.argmax(axis=1)
        correct_num = torch.sum(label_digit == out_digit).cpu().item()

        self.train_epoch_correct += correct_num
        self.train_epoch_total += B
        self.log('val_loss', val_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_acc', float(correct_num) / float(B), on_step=True, on_epoch=False, prog_bar=True)

        return val_loss

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        # Same with validation_step except for the log verbose info
        test_input, test_labels = batch
        B = test_labels.shape[0]
        test_out = self(test_input)
        test_loss = self.loss_function(test_out, test_labels, 'test')

        label_digit = test_labels.argmax(axis=1)
        out_digit = test_out.argmax(axis=1)
        correct_num = torch.sum(label_digit == out_digit).cpu().item()

        self.log('test_loss', test_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('test_acc', float(correct_num) / float(B), on_step=True, on_epoch=False, prog_bar=True)

        return test_loss

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

    def __calculate_loss_and_log(self, inputs, labels, loss_dict: Dict[str, Tuple[float, Callable]], stage: str):
        raw_loss_list = [func(inputs, labels) for _, func in loss_dict.values()]
        weighted_loss = [weight * raw_loss for (weight, _), raw_loss in zip(loss_dict.values(), raw_loss_list)]
        for name, raw_loss in zip(loss_dict.keys(), raw_loss_list):
            self.log(f'{stage}_{name}', raw_loss.item(), on_step=False, on_epoch=True, prog_bar=False)

        return sum(weighted_loss)

    def __configure_loss(self):
        # User-defined function list. Recommend using `_loss` suffix in loss names.
        loss_dict = {
            'cross_entropy_loss': (1.0, cross_entropy_loss),
        }

        def loss_func(inputs, labels, stage):
            return self.__calculate_loss_and_log(
                inputs=inputs,
                labels=labels,
                loss_dict=loss_dict,
                stage=stage
            )

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
