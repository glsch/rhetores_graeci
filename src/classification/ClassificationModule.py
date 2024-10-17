import enum
import os
from typing import Union, List, Dict, Type

from jsonargparse.typing import NonNegativeInt, NonNegativeFloat,ClosedUnitInterval, restricted_number_type, PositiveInt
from jsonargparse import lazy_instance
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable
import torch
from transformers import SchedulerType, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers
import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics.functional.classification import (multiclass_f1_score, multiclass_recall, multiclass_accuracy, multiclass_precision, multiclass_confusion_matrix)
from torchmetrics.classification import MulticlassF1Score

import numpy as np


from src.logger_config import logger

# todo: confusion matrix for each epoch
# todo: attribution of the AR chapters
# todo: stats for logical divisions of the AR

class ClassificationModule(LightningModule):
    def __init__(
            self,
            task: str,
            base_transformer: str,
            model_class: Type[AutoModelForSequenceClassification],# = transformers.models.auto.AutoModelForSequenceClassification,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            num_labels: NonNegativeInt = None,
            id2label: Dict[int, str] = None,
            optimizer: OptimizerCallable = lambda p: torch.optim.AdamW(p),
            scheduler_type: SchedulerType = SchedulerType.LINEAR,
            num_warmup_steps: NonNegativeInt = 0,
            push_to_hub: bool = False,
            confidence_threshold: NonNegativeFloat = 0.8,
    ):
        super().__init__()
        self.task = task
        self.tokenizer = tokenizer
        self.model_class = model_class
        self.num_labels = num_labels
        self.id2label = id2label

        self.push_to_hub = push_to_hub

        self.optimizer_callable = optimizer
        self.lr_scheduler_type = scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.confidence_threshold = confidence_threshold
        self.base_transformer = base_transformer

        self.model = self.model_class.from_pretrained(pretrained_model_name_or_path=self.base_transformer, num_labels=self.num_labels, id2label=self.id2label)

        self.save_hyperparameters(ignore=["base_transformer", "model_class"])

        self.epoch_outputs = {"train": [], "val": [], "test": []}
        self.epoch_labels = {"train": [], "val": [], "test": []}

    def forward(self, batch):
        return self.model.forward(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self._process_batch(batch, stage="train")
        return outputs.loss

    def on_train_epoch_end(self) -> None:
        logger.debug(f"ClassificationModule.on_train_epoch_end()")
        with torch.no_grad():
            self.compute_metrics(stage="train")

    def validation_step(self, batch, batch_idx):
        outputs = self._process_batch(batch, stage="val")
        return outputs.loss

    def on_validation_epoch_end(self) -> None:
        logger.debug(f"ClassificationModule.on_validation_epoch_end()")
        with torch.no_grad():
            self.compute_metrics(stage="val")

    def test_step(self, batch, batch_idx):
        outputs = self._process_batch(batch, stage="test")
        return outputs.loss

    def on_test_epoch_end(self) -> None:
        logger.debug(f"ClassificationModule.on_test_epoch_end()")
        with torch.no_grad():
            self.compute_metrics(stage="test")

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
    
    def compute_metrics(self, stage="train"):
        """
        A helper to compute accuracy metrics for classification.
        :param stage: The stage.
        :type stage:  str
        :return:
        """
        logger.debug(f"ClassificationModule.compute_metrics() -- Concatenating epoch logits and labels")
        save_dir = self.trainer.logger.save_dir

        if save_dir is None:
            save_dir = self.trainer.default_root_dir

        logger.debug(f"ClassificationModule.compute_metrics() -- Saving the plot to {save_dir}")

        all_logits = torch.cat(self.epoch_outputs[stage], dim=0)
        all_labels = torch.cat(self.epoch_labels[stage], dim=0)
        logger.info(f"Documents processed: {all_logits.shape[0]}")

        # if we are not in the test stage, we ignore the "<UNK>" label, as we only train on "real" classes
        # UNK will be implemented as a separate class in the test when it'll be defined
        # as the model's insufficient confidence

        ignore_index = -100
        class_labels = [self.id2label[i] for i in range(len(self.id2label))]
        # if we are testing, we need to preserve the "<UNK>" label
        if stage in ("test"):
            ignore_index = None
            class_labels = [self.id2label[i] for i in range(len(self.id2label))] + ["<UNK>"]
            all_labels[all_labels == -100] = self.num_labels

        logger.debug(f"ClassificationModule.compute_metrics() -- Predicting...")
        predictions = self.make_predictions(logits=all_logits, target=all_labels, stage=stage, confidence_threshold=None)

        logger.debug(f"ClassificationModule.compute_metrics() -- Transferring results to CPU...")
        predictions = predictions.cpu().detach()

        for i in range(predictions.shape[0]):
            logger.debug(f"Document {i} -- Prediction: {predictions[i]}")

        labels = all_labels.cpu().detach()

        # metrics
        # todo: consider using metric collection
        # however, functional interface is more tolerant to the dynamic number of classes, as in our case
        logger.debug(f"ClassificationModule.compute_metrics() -- Computing accuracy metrics...")
        f1 = multiclass_f1_score(predictions, labels, num_classes=self.num_labels, ignore_index=ignore_index)
        accuracy = multiclass_accuracy(predictions, labels, num_classes=self.num_labels, ignore_index=ignore_index)
        precision = multiclass_precision(predictions, labels, num_classes=self.num_labels, ignore_index=ignore_index)
        recall = multiclass_recall(predictions, labels, num_classes=self.num_labels, ignore_index=ignore_index)
        logger.debug(f"ClassificationModule.compute_metrics() -- Accuracy metrics: F1 {f1}, accuracy {accuracy}, precision {precision}, recall {recall}")

        ##########################
        ### Per class F1 score
        #########################

        # todo: consider adding others, too
        mcls_f1 = MulticlassF1Score(num_classes=self.num_labels, ignore_index=ignore_index, average=None)
        mcls_f1.update(predictions, labels)

        # plot F1 per class barchart
        # sorting by F1 score (?)
        f1_per_class = mcls_f1.compute().detach().cpu().numpy()
        sorted_indices = np.argsort(f1_per_class)
        sorted_f1_per_class = f1_per_class[sorted_indices]
        sorted_class_labels = np.array(class_labels)[sorted_indices]
        fig, ax = plt.subplots(figsize=(25, 25))
        bars = ax.bar(sorted_class_labels, sorted_f1_per_class)
        for bar, f1_score in zip(bars, sorted_f1_per_class):
            height = bar.get_height()
            ax.annotate(f'{f1_score:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=12)
        ax.set_xlabel("Classes", fontsize=20)
        ax.set_ylabel("F1 Score", fontsize=20)
        ax.set_xticks(range(self.num_labels))
        ax.set_xticklabels(sorted_class_labels, rotation=90, ha='right', fontsize=18)
        ax.set_title(f"F1 Score per author ('{stage}')", fontsize=25)
        ax.set_ylim(0, 1.0)

        img_path = os.path.join(save_dir, f"f1_class_{stage}.png")
        plt.savefig(img_path)
        plt.close()

        self.trainer.logger.log_image(f"f1_class_{stage}", images=[img_path], step=self.current_epoch)

        ######################
        ## Confusion matrix
        ######################
        conf_matrix = multiclass_confusion_matrix(predictions, labels, num_classes=self.num_labels, ignore_index=ignore_index)
        conf_matrix = conf_matrix.cpu().detach().numpy()

        logger.debug(f"ClassificationModule.compute_metrics() -- Confusion matrix: {conf_matrix}")
        logger.debug(f"ClassificationModule.compute_metrics() -- Creating a plot...")

        # plot confusion matrix
        plt.figure(figsize=(25, 25))
        sns.heatmap(conf_matrix, annot=True, fmt='d',
                    cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)

        plt.xlabel('Predicted', fontsize=20)
        plt.ylabel('True', fontsize=20)
        plt.title(f'Confusion Matrix ({stage.upper()})', fontsize=25)
        plt.xticks(rotation=90, fontsize=18)
        plt.yticks(fontsize=18)

        save_dir = self.trainer.logger.save_dir
        if save_dir is None:
            save_dir = self.trainer.default_root_dir
        logger.debug(
            f"ClassificationModule.compute_metrics() -- Saving the plot to {save_dir}")
        img_path = os.path.join(save_dir, f"confusion_matrix_{stage}_epoch_{self.current_epoch}.png")
        plt.savefig(img_path)
        plt.close()

        self.trainer.logger.log_image(f"confusion_matrix_{stage}", images=[img_path], step=self.current_epoch)

        # log everything
        self.log(f"{stage}/f1", f1, logger=True, on_epoch=True)
        self.log(f"{stage}/accuracy", accuracy, logger=True, on_epoch=True)
        self.log(f"{stage}/precision", precision, logger=True, on_epoch=True)
        self.log(f"{stage}/recall", recall, logger=True, on_epoch=True)

        self.epoch_outputs[stage] = []
        self.epoch_labels[stage] = []
        # reset MulticlassF1Score
        mcls_f1.reset()

    def _process_batch(self, batch, stage="train") -> SequenceClassifierOutput:
        assert self.num_labels is not None, "Number of classes must be set before processing a batch"
        logger.debug(f"ClassificationModule._process_batch() -- Processing batch for stage: {stage}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch: {batch}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch input_ids: {batch['input_ids'].shape}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch labels: {batch['labels'].shape}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch labels: {batch['labels']}")
        logger.debug(f"ClassificationModule._process_batch() -- Num classes: {self.num_labels}")

        classifier_output = self.forward(batch)
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
            reject_label = self.num_labels

        logger.debug(f"ClassificationModule.make_predictions() -- Reject label: {reject_label} (stage: {stage})")
        logger.debug(f"ClassificationModule.make_predictions() -- Getting probabilities...")

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        logger.debug(f"ClassificationModule.make_predictions() -- Probabilities: {probabilities}")

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



