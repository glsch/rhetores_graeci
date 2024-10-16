import argparse
import os
from typing import Any, Tuple

from lightning.pytorch import LightningModule

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import (
    BertConfig,
    BertModel,
    AutoConfig,
    BertForMaskedLM,
    PretrainedConfig,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    get_scheduler,
    AutoTokenizer,
    SchedulerType
)

from jsonargparse import lazy_instance

from lightning.pytorch.cli import OptimizerCallable

from jsonargparse.typing import NonNegativeInt, NonNegativeFloat,ClosedUnitInterval, restricted_number_type, PositiveInt

from src.path_manager import PathManager
from src.logger_config import logger

class AutoModelForMaskedLMWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = "bowphs/GreBerta"):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if isinstance(pretrained_model_name_or_path, str):
            self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path)

    def forward(self, x):
        return self.model(x)

class MlmTuningModule(LightningModule):

    def __init__(
            self,
            model: torch.nn.Module = lazy_instance(AutoModelForMaskedLMWrapper,
                                                   pretrained_model_name_or_path="bowphs/GreBerta"),
            scheduler_type: SchedulerType = SchedulerType.LINEAR,
            optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(p),
            weight_decay: NonNegativeFloat = 0.0,
            num_warmup_steps: NonNegativeInt = 0,
    ):
        super().__init__()

        self.model = model
        if isinstance(model, AutoModelForMaskedLMWrapper):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.pretrained_model_name_or_path)
        else:
            raise ValueError("Model must be an instance of AutoModelForMaskedLMWrapper")

        self.weight_decay = weight_decay
        self.optimizer_callable = optimizer
        self.lr_scheduler_type = scheduler_type
        self.num_warmup_steps = num_warmup_steps

        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def _process_batch(self, batch, stage="train"):
        output = self.forward(batch)
        self.log(f"{stage}/loss", output.loss, sync_dist=True if self.trainer.num_devices > 1 else False)
        return output

    def training_step(self, batch, batch_idx):
        return self._process_batch(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._process_batch(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._process_batch(batch, "test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"] # layers that should not be decayed, as in the original Huggingface training script
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.optimizer_callable(optimizer_grouped_parameters)

        num_training_steps = self.trainer.estimated_stepping_batches

        logger.debug(f"ModelForTransferMLM.configure_optimizers() -- Warmup steps: {self.num_warmup_steps}, {type(self.num_warmup_steps)}")
        logger.debug(f"ModelForTransferMLM.configure_optimizers() -- Num training steps: {num_training_steps}, {type(num_training_steps)}")

        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return [optimizer], [lr_scheduler]