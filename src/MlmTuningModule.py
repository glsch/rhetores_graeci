import argparse
import os
from typing import Any, Tuple, Type, Union, Dict

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
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# todo: add pushing to hub

class MlmTuningModule(LightningModule):

    def __init__(
            self,
            task: str,
            base_transformer: str,
            model_class: Type[AutoModelForMaskedLM],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(p),
            scheduler_type: SchedulerType = SchedulerType.LINEAR,
            num_warmup_steps: NonNegativeInt = 0,
            push_to_hub: bool = False,
            **kwargs
    ):
        super().__init__()

        self.task = task
        self.tokenizer = tokenizer
        self.model_class = model_class

        self.push_to_hub = push_to_hub

        self.optimizer_callable = optimizer
        self.lr_scheduler_type = scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.base_transformer = base_transformer

        self.model = self.model_class.from_pretrained(pretrained_model_name_or_path=self.base_transformer)

        self.save_hyperparameters(ignore=["base_transformer", "model_class"])

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
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.optimizer_callable(optimizer_grouped_parameters)

        logger.info(f"MlmTuningModule.configure_optimizers() -- Using optimizer: {optimizer}, {type(optimizer)}, {optimizer.__class__.__name__}")

        num_training_steps = self.trainer.estimated_stepping_batches

        logger.info(f"ModelForTransferMLM.configure_optimizers() -- Warmup steps: {self.num_warmup_steps}, {type(self.num_warmup_steps)}")
        logger.info(f"ModelForTransferMLM.configure_optimizers() -- Num training steps: {num_training_steps}, {type(num_training_steps)}")
        logger.info(f"ModelForTransferMLM.configure_optimizers() -- Num processes: {os.cpu_count()}")

        scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                #"monitor": "val/loss",
                #"strict": False,
                "name": self.lr_scheduler_type
            },
        }