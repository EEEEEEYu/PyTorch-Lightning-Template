# A Convenient Template To Fully Utilize PyTorch
### Important Note

This repository is based on https://github.com/miracleyoo/pytorch-lightning-template.

Credit to the original author https://github.com/miracleyoo

Several improvements were made based on the previous repository
- Removed unnecessary packages, functions and arguments. This repository now uses minimal external package.
- Fixed compatibility issues and removed deprecated functions. Now it's compatible with PyTorch Lightning 2.0+.
- Renamed functions, variables to avoid confusions. The repository is made easy to understand with extensive comments.
- Refactored several classes and main function to enhance generalization capability.

### About PyTorch Lightning (e.g. Lightning AI)

#### Official Website
https://lightning.ai/docs/pytorch/stable/

#### How It Works
PyTorch Lightning is essentially a higher level wrapper for the PyTorch framework. It provides two basic components:
`Trainer` and `LightningModule` for users to customize their model training/testing progress.
- LightningModule: This class abstracts the most crucial steps in a PyTorch workflow into hook functions
(e.g. configure_optimizers, train_step). Users need to override these hook functions to define their customized workflow logics.
These hook functions help users to organize their workflow logics and conveniently utilize functionalities like
checkpoint, logging, learning rate scheduler and distributed training.
- Once you’ve organized your PyTorch code into a LightningModule, the Trainer automates everything else. You can pass
various arguments to trainer to control its automation progress. Your overridden hook functions will be executed
in a best-practice style.

### How To Use
1. Define your own torch.utils.data.Dataset class `MyDataset`, and save a `my_dataset.py` file in the data folder.
Note that your class name `MyDataset`, which is **CamelCase**, should contain the same sequential words as in the **snake_case**
`my_dataset.py` file. The data interface will instantiate your dataset class using the file name.
2. Define your own torch.nn.Module class `MyModel`, and save a `my_model.py` file in the model folder. The model interface
will instantiate your model using the file name.
3. Customize your loss functions in the `model_interface.py` file.
4. Configure your hyperparameters in the main function. Note that `LightningModule` and `LightningDataModule` utilizes
`self.save_hyperparameters()` method to export arguments to `self.hparams`. But your dataset and model extract arguments
from `**kwargs`(expanded from `self.hparams`). So make sure your required arguments in model and dataset have the same
name as in the main function.
5. Launch the pipeline!
