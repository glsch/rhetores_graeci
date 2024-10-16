import enum
from typing import Union, List

from jsonargparse.typing import NonNegativeInt, NonNegativeFloat,ClosedUnitInterval, restricted_number_type, PositiveInt
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable
import torch
from transformers import SchedulerType, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer
from transformers import RobertaForSequenceClassification

# todo: confusion matrix for each epoch
# todo: attribution of the AR chapters
# todo: stats for logical divisions of the AR

class AutoModelForSequenceClassificationWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = "bowphs/GreBerta"):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if isinstance(pretrained_model_name_or_path, str):
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path)
            if self.pretrained_model_name_or_path != "altsoph/bert-base-ancientgreek-uncased":
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

    def forward(self, x):
        return self.model(x)

class ClassificationModule(LightningModule):
    def __init__(
            self,
            model: torch.nn.Module = lazy_instance(AutoModelForSequenceClassificationWrapper, pretrained_model_name_or_path="bowphs/GreBerta"),
            optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(p),
            scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU,
            num_warmup_steps: NonNegativeInt = 0,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = None

        if isinstance(model, AutoModelForSequenceClassificationWrapper):
            self.tokenizer = self.model.tokenizer
        elif hasattr(model, "transformer"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.transformer.config._name_or_path)
        else:
            raise ValueError("Model must have a 'transformer' attribute or be an instance of AutoModelForSequenceClassificationWrapper")

        self.optimizer_callable = optimizer
        self.scheduler_type = scheduler_type

        self.num_warmup_steps = num_warmup_steps

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        classifier_output = self.model(**batch)
        loss = classifier_output.loss

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        params_list = [self.transformer.parameters(), self.classification_head.parameters()]
        optimizer = self.optimizer_callable(params_list)

        num_training_steps = self.trainer.estimated_stepping_batches,
        lr_scheduler = get_scheduler(
            name=self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return [optimizer], [lr_scheduler]