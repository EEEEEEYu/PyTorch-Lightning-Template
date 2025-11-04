import importlib
from dataclasses import asdict

import torch
import lightning.pytorch as pl

from loss.loss_funcs import cross_entropy_loss
from torchmetrics.functional.classification import multiclass_accuracy
from configs.sections import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    DataConfig,
)


class ModelInterface(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
        training_cfg: TrainingConfig,
        data_cfg: DataConfig,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg
        self.num_classes = self.data_cfg.dataset.num_classes

        self.save_hyperparameters(
            {
                "model": asdict(self.model_cfg),
                "optimizer": asdict(self.optimizer_cfg),
                "scheduler": asdict(self.scheduler_cfg),
                "training": asdict(self.training_cfg),
                "data": asdict(self.data_cfg),
            }
        )

        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()

    def forward(self, x):
        return self.model(x)

    # For all these hook functions like on_XXX_<epoch|batch>_<end|start>(),
    # check document: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
    # Epoch level training logging
    def on_train_epoch_end(self):
        pass

    # Caution: self.model.train() is invoked
    # For logging, check document: https://lightning.ai/docs/pytorch/stable/extensions/logging.html#automatic-logging
    # Important clarification for new users:
    # 1. If on_step=True, a _step suffix will be concatenated to metric name. Same for on_epoch, but epoch-level metrics will be automatically averaged using batch_size as weight.
    # 2. If enable_graph=True, .detach() will not be invoked on the value of metric. Could introduce potential error.
    # 3. If sync_dist=True, logger will average metrics across devices. This introduces additional communication overhead, and not suggested for large metric tensors.
    # We can also define customized metrics aggregator for incremental step-level aggregation(to be merged into epoch-level metrics).
    def training_step(self, batch, batch_idx):
        train_input, train_labels = batch
        train_out_logits = self(train_input)
        train_loss = self.loss_function(train_out_logits, train_labels, 'train')

        train_step_top1_acc = multiclass_accuracy(train_out_logits, train_labels, num_classes=self.num_classes, top_k=1)
        train_step_top5_acc = multiclass_accuracy(train_out_logits, train_labels, num_classes=self.num_classes, top_k=5)
        self.log('train_top1_acc', value=train_step_top1_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=train_input.shape[0])
        self.log('train_top5_acc', value=train_step_top5_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=train_input.shape[0])

        train_step_output = {
            'loss': train_loss,
            'pred': train_out_logits,
            'ground_truth': train_labels
        }

        return train_step_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        val_input, val_labels = batch
        val_out_logits = self(val_input)
        val_loss = self.loss_function(val_out_logits, val_labels, 'val')
        val_step_top1_acc = multiclass_accuracy(val_out_logits, val_labels, num_classes=self.num_classes, top_k=1)
        val_step_top5_acc = multiclass_accuracy(val_out_logits, val_labels, num_classes=self.num_classes, top_k=5)
        self.log('val_top1_acc', value=val_step_top1_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=val_input.shape[0])
        self.log('val_top5_acc', value=val_step_top5_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=val_input.shape[0])

        val_step_output = {
            'loss': val_loss,
            'pred': val_out_logits,
            'ground_truth': val_labels
        }

        return val_step_output

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        test_input, test_labels = batch
        test_out_logits = self(test_input)
        test_loss = self.loss_function(test_out_logits, test_labels, 'test')
        test_step_top1_acc = multiclass_accuracy(test_out_logits, test_labels, num_classes=self.num_classes, top_k=1)
        test_step_top5_acc = multiclass_accuracy(test_out_logits, test_labels, num_classes=self.num_classes, top_k=5)
        self.log('test_top1_acc', value=test_step_top1_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=test_input.shape[0])
        self.log('test_top5_acc', value=test_step_top5_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=test_input.shape[0])

        test_step_output = {
            'loss': test_loss,
            'pred': test_out_logits,
            'ground_truth': test_labels
        }

        return test_step_output

    def configure_optimizers(self):
        # https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html
        try:
            optimizer_class = getattr(torch.optim, self.optimizer_cfg.name)
        except AttributeError as exc:
            raise ValueError(f"Invalid optimizer: OPTIMIZER.{self.optimizer_cfg.name}") from exc

        optimizer_arguments = dict(self.optimizer_cfg.arguments or {})
        optimizer_instance = optimizer_class(params=self.model.parameters(), **optimizer_arguments)

        learning_rate_scheduler_cfg = self.scheduler_cfg.learning_rate
        if not learning_rate_scheduler_cfg.enabled:
            return [optimizer_instance]

        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, learning_rate_scheduler_cfg.name)
        except AttributeError as exc:
            raise ValueError(
                f"Invalid learning rate scheduler: SCHEDULER.learning_rate.{learning_rate_scheduler_cfg.name}."
            ) from exc

        scheduler_arguments = dict(learning_rate_scheduler_cfg.arguments or {})
        scheduler_instance = scheduler_class(optimizer=optimizer_instance, **scheduler_arguments)

        return [optimizer_instance], [scheduler_instance]

    def __configure_loss(self):
        def loss_func(preds, labels, stage):
            CE_loss = 1.0 * cross_entropy_loss(pred=preds, gt=labels)
            self.log(f'{stage}_CE_loss', CE_loss, on_step=True, on_epoch=True, prog_bar=True)

            final_loss = CE_loss
            self.log(f'{stage}_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True)

            return final_loss

        return loss_func

    def __load_model(self):
        file_name = self.model_cfg.file_name
        class_name = self.model_cfg.class_name
        if class_name is None:
            raise ValueError("MODEL.class_name must be specified in the configuration.")
        if file_name is None:
            raise ValueError("MODEL.file_name must be specified in the configuration.")
        try:
            model_class = getattr(importlib.import_module('model.' + file_name, package=__package__), class_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {file_name}.{class_name}!')

        model_init_kwargs = asdict(self.model_cfg)
        model_init_kwargs.pop("class_name", None)
        model_init_kwargs.pop("file_name", None)

        model = model_class(**model_init_kwargs)
        if self.training_cfg.use_compile:
            model = torch.compile(model)
        return model
