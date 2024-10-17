import enum
import os
from typing import Union, List

from jsonargparse.typing import NonNegativeInt, NonNegativeFloat,ClosedUnitInterval, restricted_number_type, PositiveInt
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable
import torch
from transformers import SchedulerType, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaForSequenceClassification

from src.logger_config import logger



# todo: confusion matrix for each epoch
# todo: attribution of the AR chapters
# todo: stats for logical divisions of the AR

class AutoModelForSequenceClassificationWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = "bowphs/GreBerta", instantiate: bool = True):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if isinstance(pretrained_model_name_or_path, str):
            if instantiate:
                self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path)
                if self.pretrained_model_name_or_path != "altsoph/bert-base-ancientgreek-uncased":
                    self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

    def forward(self, x):
        return self.model(**x)

class ClassificationModule(LightningModule):
    def __init__(
            self,
            num_classes: NonNegativeInt,
            model: torch.nn.Module = lazy_instance(AutoModelForSequenceClassificationWrapper, pretrained_model_name_or_path="bowphs/GreBerta"),
            optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(p),
            scheduler_type: SchedulerType = SchedulerType.LINEAR,
            num_warmup_steps: NonNegativeInt = 0,
            push_to_hub: bool = False,
            confidence_threshold: NonNegativeFloat = 0.8,
    ):
        super().__init__()

        self.model = model
        self.model.model.config.num_labels = num_classes

        self.push_to_hub = push_to_hub

        self.tokenizer = self.model.tokenizer

        self.optimizer_callable = optimizer
        self.lr_scheduler_type = scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.confidence_threshold = confidence_threshold

        self.save_hyperparameters()

        self.epoch_outputs = {"train": [], "val": [], "test": []}
        self.epoch_labels = {"train": [], "val": [], "test": []}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self._process_batch(batch, stage="train")
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self._process_batch(batch, stage="val")
        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self._process_batch(batch, stage="test")
        return outputs.loss

    def configure_optimizers(self):
        no_decay = ["bias",
                    "LayerNorm.weight"]  # layers that should not be decayed, as in the original Huggingface training script
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

        logger.info(
            f"ClassificationModule.configure_optimizers() -- Using optimizer: {optimizer}, {type(optimizer)}, {optimizer.__class__.__name__}")

        num_training_steps = self.trainer.estimated_stepping_batches

        logger.info(
            f"ClassificationModule.configure_optimizers() -- Warmup steps: {self.num_warmup_steps}, {type(self.num_warmup_steps)}")
        logger.info(
            f"ClassificationModule.configure_optimizers() -- Num training steps: {num_training_steps}, {type(num_training_steps)}")
        logger.info(f"ClassificationModule.configure_optimizers() -- Num processes: {os.cpu_count()}")

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
                # "monitor": "val/loss",
                # "strict": False,
                "name": self.lr_scheduler_type
            },
        }

    def _process_batch(self, batch, stage="train") -> SequenceClassifierOutput:
        classifier_output = self.model(**batch)
        self.log(f"{stage}/loss", classifier_output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # saving for epoch
        classifier_output["labels"] = batch["labels"]
        self.epoch_outputs[stage].append(classifier_output.logits)
        self.epoch_labels[stage].append(classifier_output.labels)

        return classifier_output

    def make_predictions(self, logits, target=None, stage="train", confidence_threshold: float = None):
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        logger.debug(f"ClassificationModule.make_predictions() -- Making predictions")
        # since metrics do not allow to have -1 as a label, we need to replace it with the number of heads,
        # which always equals to the max index in self.heads + 1

        reject_label = -100
        if stage in ("test"):
            reject_label = self.num_classes

        logger.debug(
            f"ClassificationModule.make_predictions() -- Reject label: {reject_label} (stage: {stage})")

        logger.debug(
            f"ClassificationModule.make_predictions() -- Getting probabilities...")

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        logger.debug(
            f"ClassificationModule.make_predictions() -- Probabilities: {probabilities}")

        if not stage in ("test", "predict"):
            return torch.argmax(probabilities, dim=1)

        # if the difference between the top two probabilities is less than the threshold,
        # we do not predict
        # (consistently with PAN 2019 and our own baselines)
        rejection_threshold = self.confidence_threshold
        method = "pt"
        if method == "pt":
            top_probs, top_indices = torch.max(probabilities, dim=1)
            # Compare the top probability with the rejection threshold
            predictions = torch.where(top_probs > 0.80, top_indices, reject_label)

        elif method == "difference":
            logger.debug(
                f"ClassificationModule.make_predictions() -- Applying threshold: {rejection_threshold}")
            top2_probs, top2_indices = torch.topk(probabilities, 2, dim=1)
            diff = top2_probs[:, 0] - top2_probs[:, 1]
            # if no prediction is made -1 is returned, which corresponds to the "<UNK>" label in the test data
            predictions = torch.where(diff > rejection_threshold, top2_indices[:, 0], reject_label)
        else:
            raise ValueError(f"Unknown prediction method {method}")
        logger.debug(
            f"ClassificationModule.make_predictions() -- Predictions: {predictions}")

        return predictions



