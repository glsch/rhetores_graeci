import argparse
import os
from typing import Any, Tuple, Type

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

# todo: add pushing to hub

class AutoModelForMaskedLMWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = "bowphs/GreBerta"):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if isinstance(pretrained_model_name_or_path, str):
            self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path)
            if self.pretrained_model_name_or_path != "altsoph/bert-base-ancientgreek-uncased":
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

    def forward(self, x):
        return self.model(**x)

class MlmTuningModule(LightningModule):

    def __init__(
            self,
            scheduler_type: SchedulerType = SchedulerType.LINEAR,
            model: torch.nn.Module = lazy_instance(AutoModelForMaskedLMWrapper,
                                                   pretrained_model_name_or_path="bowphs/GreBerta"),
            optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(p),
            num_warmup_steps: NonNegativeInt = 0,
            push_to_hub: bool = False
    ):
        super().__init__()

        self.model = model
        if isinstance(model, AutoModelForMaskedLMWrapper):
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError("Model must be an instance of AutoModelForMaskedLMWrapper")

        self.optimizer_callable = optimizer
        self.lr_scheduler_type = scheduler_type
        self.num_warmup_steps = num_warmup_steps

        self.push_to_hub = push_to_hub

        self._dir_save_path = None

        self.save_hyperparameters()

    @property
    def dir_save_path(self):
        if self._dir_save_path is None:
            self._dir_save_path = self._get_save_dir()

        return self._dir_save_path

    def _get_save_dir(self):
        save_dir = self.trainer.logger.save_dir
        if save_dir is None:
            save_dir = self.trainer.default_root_dir

        return save_dir

    def forward(self, batch):
        return self.model(batch)

    def _process_batch(self, batch, stage="train"):
        output = self.forward(batch)
        self.log(f"{stage}/loss", output.loss, sync_dist=True if self.trainer.num_devices > 1 else False)
        return output

    def training_step(self, batch, batch_idx):
        return self._process_batch(batch, "train")

    # def on_train_epoch_end(self) -> None:
    #     assert isinstance(self.model, AutoModelForMaskedLMWrapper)
    #     # huggingface model saving
    #     path = os.path.join(self.dir_save_path, "huggingface")
    #     self.model.tokenizer.save_pretrained(path)
    #     self.model.model.save_pretrained(path, from_pt=True, safe_serialization=False)
    #
    #     # pushing to hub, if necessary
    #     if self.push_to_hub:
    #         api.upload_folder(
    #             commit_message=f"Pushing after epoch {self.trainer.current_epoch}",
    #             folder_path=path,
    #             repo_id=repo_id,
    #             repo_type="model",
    #             token=hub_token,
    #             ignore_patterns=[".gitattributes", ".gitignore", "**/checkpoints/**", "logs/**", "**/wandb/**"],
    #             delete_patterns=[".gitattributes", ".gitignore", "**/checkpoints/**", "logs/**", "**/wandb/**"]
    #         )
    #
    #         model.model.push_to_hub(repo_id=repo_id, token=hub_token)

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
                # "weight_decay": self.weight_decay,
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

        # lr_scheduler = get_scheduler(
        #     name=self.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=self.num_warmup_steps,
        #     num_training_steps=num_training_steps
        # )
        #
        # logger.info(f"MlmTuningModule.configure_optimizers() -- Using scheduler: {lr_scheduler}, {type(lr_scheduler)}, {lr_scheduler.__class__.__name__}")
        #
        # return [optimizer], [lr_scheduler]