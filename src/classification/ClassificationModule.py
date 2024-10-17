import enum
import os
from typing import Union, List, Dict, Type, Any

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
from tqdm import tqdm
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from torchmetrics.functional.classification import (multiclass_f1_score, multiclass_recall, multiclass_accuracy, multiclass_precision, multiclass_confusion_matrix)
from torchmetrics.classification import MulticlassF1Score, MulticlassCalibrationError
from torch.nn import CrossEntropyLoss

import numpy as np
import pandas as pd

from src.logger_config import logger

# todo: attribution of the AR chapters
# todo: stats for logical divisions of the AR

def _ce_plot(self, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)

    conf = dim_zero_cat(self.confidences)
    acc = dim_zero_cat(self.accuracies)
    bin_width = 1 / self.n_bins

    bin_ids = torch.round(
        torch.clamp(conf * self.n_bins, 1e-5, self.n_bins - 1 - 1e-5)
    )
    val, inverse, counts = bin_ids.unique(
        return_inverse=True, return_counts=True
    )
    counts = counts.float()
    val_oh = torch.nn.functional.one_hot(
        val.long(), num_classes=self.n_bins
    ).float()

    # add 1e-6 to avoid division NaNs
    values = (
            val_oh.T
            @ torch.sum(
        acc.unsqueeze(1) * torch.nn.functional.one_hot(inverse).float(),
        0,
    )
            / (val_oh.T @ counts + 1e-6)
    )

    plt.rc("axes", axisbelow=True)
    ax.hist(
        x=[bin_width * i * 100 for i in range(self.n_bins)],
        weights=values.cpu() * 100,
        bins=[bin_width * i * 100 for i in range(self.n_bins + 1)],
        alpha=0.7,
        linewidth=1,
        edgecolor="#0d559f",
        color="#1f77b4",
    )

    ax.plot([0, 100], [0, 100], "--", color="#0d559f")
    plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_xlabel("Top-class Confidence (%)", fontsize=16)
    ax.set_ylabel("Success Rate (%)", fontsize=16)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")
    if fig is not None:
        fig.tight_layout()
    return fig, ax

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
            init_temp: float = 1.0,
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

        self.epoch_outputs = {"train": [], "val": [], "test": [], "predict": []}
        self.epoch_labels = {"train": [], "val": [], "test": [], "predict": []}

        # an ugly workaround to keep predictions here, too
        self.sigla = []

        self.set_temperature(val=init_temp)

        self.calibrated = False
        
    def on_test_start(self):
        calibration_dataloader = self.trainer.datamodule.val_dataloader()

        with torch.inference_mode(False):
            self.train(mode=True)
            self.temperature[0].requires_grad = True
            self.temperature_calibration(calibration_dataloader)
            
    def temperature_calibration(self, calibration_dataloader):
        """
        Calibrate the model by optimizing a vector which is applied to the model's logits.
        :param calibration_dataloader:
        :return:
        """
        # todo: consider adding a callback to the trainer to save the calibration plot
        # todo: consider keeping UNK in the labels for calibration
        optimizer = torch.optim.LBFGS(
            self.temperature, lr=0.15, max_iter=1000000
        )

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(calibration_dataloader, disable=False):
                model_outputs = self(batch)
                logits_list.append(model_outputs.logits)
                labels_list.append(batch.labels)

        all_logits = torch.cat(logits_list).detach().to(self.device)
        all_labels = torch.cat(labels_list).detach().to(self.device)

        logger.debug(f"Temperature: {self.temperature[0]}")
        #logger.debug(f"Temperature-bias: {self.temperature[1]}")

        MulticlassCalibrationError.plot = _ce_plot
        ce = MulticlassCalibrationError(num_classes=self.num_labels, n_bins=20, ignore_index=-100)
        ce.update(torch.softmax(all_logits, dim=1), all_labels)
        self.log("test/mce_bc", ce.compute(), logger=True, on_step=False, on_epoch=True)
        logger.info(f"Calibration error (before calibration): {ce.compute()}")

        fig, ax = plt.subplots(figsize=(25, 25))
        ce.plot(ax=ax)

        save_dir = self.trainer.logger.save_dir
        if save_dir is None:
            save_dir = self.trainer.default_root_dir

        logger.debug(f"MultiheadedClassifier.vector_calibration() -- Saving the plot to {save_dir}")

        img_path = os.path.join(save_dir, f"top_label_confidence_vector_ac.png")
        plt.savefig(img_path)
        plt.close(fig)

        self.trainer.logger.log_image(f"top_label_confidence_vector_bc", images=[img_path])

        criterion = CrossEntropyLoss(ignore_index=-100).requires_grad_(True)

        def calib_eval() -> float:
            optimizer.zero_grad()
            loss = criterion(self._scale(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(calib_eval)

        self.calibrated = True

        logger.debug(f"Calibrated temperature {self.temperature}")
        #logger.debug(f"Calibrated temperature-bias: {self.temperature[1]}")

        ce.reset()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in tqdm(calibration_dataloader, disable=False):
                model_outputs = self(batch)
                logits_list.append(model_outputs.logits)
                labels_list.append(batch.labels)

        all_logits = torch.cat(logits_list).detach().to(self.device)
        all_labels = torch.cat(labels_list).detach().to(self.device)

        ce.update(torch.softmax(self._scale(all_logits), dim=1), all_labels)
        self.log("test/mce_ac", ce.compute(), logger=True, on_step=False, on_epoch=True)
        logger.info(f"Calibration error (after calibration): {ce.compute()}")

        fig, ax = plt.subplots(figsize=(25, 25))
        ce.plot(ax=ax)

        save_dir = self.trainer.logger.save_dir
        if save_dir is None:
            save_dir = self.trainer.default_root_dir

        logger.debug(f"MultiheadedClassifier.vector_calibration() -- Saving the plot to {save_dir}")

        img_path = os.path.join(save_dir, f"top_label_confidence_vector_ac.png")
        plt.savefig(img_path)
        plt.close(fig)

        self.trainer.logger.log_image(f"top_label_confidence_vector_ac", images=[img_path])

    def forward(self, batch):
        return self.model.forward(**batch.to(self.device))

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

    def predict_step(self, batch) -> Any:
        outputs = self._process_batch(batch, stage="predict")
        return outputs.loss

    def on_predict_epoch_end(self) -> None:
        logger.debug(f"ClassificationModule.on_predict_epoch_end()")
        with torch.no_grad():
            self.get_per_chapter_stats(stage="predict")

    def get_per_chapter_stats(self, stage="predict"):
        path = self.trainer.logger.experiment.dir
        if path is None:
            path = self.trainer.default_root_dir
        save_dir = self.trainer.logger.save_dir if self.trainer.logger.save_dir is not None else self.trainer.default_root_dir
        logger.debug(f"ClassificationModule.compute_metrics() -- Saving the plot to {save_dir}")

        all_logits = torch.cat(self.epoch_outputs[stage], dim=0).cpu().detach()
        all_sigla = torch.cat(self.sigla, dim=0).cpu().detach()

        sorted_sigla, indices = torch.sort(all_sigla, dim=0)
        sorted_logits = all_logits[indices]

        probabilities = torch.nn.functional.softmax(sorted_logits, dim=1)

        sigla_np = sorted_sigla.cpu().numpy()
        probs_np = probabilities.cpu().numpy()

        df = pd.DataFrame(probs_np)
        df['siglum'] = sigla_np

        df_melted = df.melt(id_vars=['siglum'], var_name='class', value_name='probability')

        df_grouped = df_melted.groupby(['siglum', 'class'])['probability'].mean().reset_index()
        df_final = df_grouped.pivot(index='siglum', columns='class', values='probability').reset_index()

        df_final.columns.name = None
        df_final = df_final.rename(columns={col: self.id2label[col] for col in df_final.columns if col != 'siglum'})

        #df_final = df_final.assign(author_name=df_final['class'].apply(lambda x: self.id2label[x]))

        results = pd.DataFrame()
        for chapter in df_final["siglum"].unique().tolist():
            top5 = df_final[df_final["siglum"] == chapter].melt(id_vars=["siglum"], var_name="class", value_name="probability").sort_values(by="probability", ascending=False).head(5)
            results = pd.concat([results, top5])

        results.to_csv(os.path.join(path, f"chapter_predictions_{self.base_transformer.replace('/', '_')}.csv"), index=False)

        return results

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
        f1 = multiclass_f1_score(predictions, labels, num_classes=len(class_labels), ignore_index=ignore_index)
        accuracy = multiclass_accuracy(predictions, labels, num_classes=len(class_labels), ignore_index=ignore_index)
        precision = multiclass_precision(predictions, labels, num_classes=len(class_labels), ignore_index=ignore_index)
        recall = multiclass_recall(predictions, labels, num_classes=len(class_labels), ignore_index=ignore_index)
        logger.debug(f"ClassificationModule.compute_metrics() -- Accuracy metrics: F1 {f1}, accuracy {accuracy}, precision {precision}, recall {recall}")

        ##########################
        ### Per class F1 score
        #########################

        # todo: consider adding others, too
        mcls_f1 = MulticlassF1Score(num_classes=len(class_labels), ignore_index=ignore_index, average=None)
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
        ax.set_xticks(range(len(class_labels)))
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
        conf_matrix = multiclass_confusion_matrix(predictions, labels, num_classes=len(class_labels), ignore_index=ignore_index)
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

    def set_temperature(self, val: float) -> None:
        """Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if val <= 0:
            raise ValueError("Temperature value must be positive.")

        self.temp = torch.nn.Parameter(
            torch.ones(1, device=self.device) * val, requires_grad=True
        )

    @property
    def temperature(self) -> list:
        return [self.temp]

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale the prediction with the optimal temperature.

        Args:
            logits (Tensor): logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """
        return logits / self.temperature[0]

    def _process_batch(self, batch, stage="train") -> SequenceClassifierOutput:
        assert self.num_labels is not None, "Number of classes must be set before processing a batch"
        logger.debug(f"ClassificationModule._process_batch() -- Processing batch for stage: {stage}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch: {batch}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch input_ids: {batch['input_ids'].shape}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch labels: {batch['labels'].shape}")
        logger.debug(f"ClassificationModule._process_batch() -- Batch labels: {batch['labels']}")
        logger.debug(f"ClassificationModule._process_batch() -- Num classes: {self.num_labels}")


        if stage == "predict":
            self.sigla.append(batch["siglum"])
            del batch["siglum"]

        classifier_output = self.forward(batch)

        if stage != "predict":
            self.log(f"{stage}/loss", classifier_output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # saving for epoch
        classifier_output["labels"] = batch["labels"]

        logits2save = classifier_output.logits.clone().detach()
        labels2save = classifier_output.labels.clone().detach()

        if self.calibrated:
            logits2save = self._scale(logits2save)

        # self.save_logits(
        #     logits=logits2save,
        #     labels=batch.target.clone().detach(),
        #     stage=stage
        # )

        self.epoch_outputs[stage].append(logits2save)
        self.epoch_labels[stage].append(labels2save)

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
        rejection_threshold = confidence_threshold
        method = "pt"
        if method == "pt":
            logger.debug(f"ClassificationModule.make_predictions() -- Applying threshold: {rejection_threshold}")
            top_probs, top_indices = torch.max(probabilities, dim=1)
            logger.debug(f"ClassificationModule.make_predictions() -- Top probs: {top_probs}, Top indices: {top_indices}")
            # Compare the top probability with the rejection threshold
            predictions = torch.where(top_probs > rejection_threshold, top_indices, reject_label)

        elif method == "difference":
            logger.debug(f"ClassificationModule.make_predictions() -- Applying threshold: {rejection_threshold}")
            top2_probs, top2_indices = torch.topk(probabilities, 2, dim=1)
            diff = top2_probs[:, 0] - top2_probs[:, 1]
            # if no prediction is made -1 is returned, which corresponds to the "<UNK>" label in the test data
            predictions = torch.where(diff > rejection_threshold, top2_indices[:, 0], reject_label)
        else:
            raise ValueError(f"Unknown prediction method {method}")
        logger.debug(
            f"ClassificationModule.make_predictions() -- Predictions: {predictions}")

        return predictions



