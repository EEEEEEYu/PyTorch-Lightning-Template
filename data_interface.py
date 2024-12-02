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
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# This class inherits the LightningDataModule class, which is basically a wrapper for pytorch dataloader.
# You need to initialize the dataloader using your custom torch.utils.data.Dataset class.
class DataInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        # This save_hyperparameter() method moves arguments to self.hparams, which are used for automatic logging
        self.save_hyperparameters()
        self.train_set, self.validation_set, self.test_set = self.__load_data_module()

    # Lightning hook function, override to implement loading train dataset
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=self.hparams.persistent_workers
        )

    # Lightning hook function, override to implement loading validation dataset
    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers
        )

    # Lightning hook function, override to implement loading test dataset
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers
        )

    # Lightning hook function. Implement as needed.
    def predict_dataloader(self):
        pass

    def __load_data_module(self):
        name = self.hparams.dataset_class_name
        # Attempt to import the `CamelCase` class name from the `snake_case.py` module. The module should be placed
        # within the same folder as model_interface.py. Always name your model file name as `snake_case.py` and
        # model class name as corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            data_class = getattr(importlib.import_module('data.' + name, package=__package__), camel_name)
        except Exception:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

        return self.__instantiate(data_class=data_class, purpose='train'), self.__instantiate(data_class=data_class, purpose='validation'), self.__instantiate(data_class=data_class, purpose='test')

    def __instantiate(self, data_class, **other_args):
        # Instantiate a Dataset class using the imported class name and parameters from self.hparams dictionary.
        # You can also input any args to overwrite the corresponding value in self.hparams.
        target_args = inspect.getfullargspec(data_class.__init__).args[1:]
        this_args = self.hparams.keys()
        merged_args = {}
        # Only assign arguments that are required in the user-defined torch.utils.data.Dataset subclass by their name.
        # You need to define the required arguments in main function.
        for arg in target_args:
            if arg in this_args:
                merged_args[arg] = getattr(self.hparams, arg)
        merged_args.update(other_args)
        return data_class(**merged_args)