# python standard modules
import os

# third-party modules
from lightning.pytorch.cli import LightningCLI
import torch
import numpy as np
import random

# project modules
from src.classification.ClassificationModule import ClassificationModule # noqa: F401
from src.datasets.AncientGreekDataModule import AncientGreekDataModule as AncientGreek
from src.MlmTuningModule import MlmTuningModule # noqa: F401

class RhetoresGraecigCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        logger_config = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            'init_args': {
                "project": "RhetoresGraeci",
                "log_model": "all"
            },
        }

        callbacks = [
            {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint"},
            {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
             "init_args": {"logging_interval": "step"}},
            {"class_path": "src.callbacks.PushToHuggingfaceCallback", "init_args": {"repo_owner": "glsch"}},
            {"class_path": "lightning.pytorch.callbacks.EarlyStopping", "init_args": {"min_delta": 0.01, "patience": 3, "monitor": "val/loss", "mode": "min", "check_on_train_epoch_end": False, "verbose": True}},
        ]

        default_epithets = [
            "Rhet.",
            "Orat."
        ]

        parser.set_defaults({
            "trainer.logger": logger_config,
            "trainer.precision": 32,
            "trainer.max_epochs": 30,
            "trainer.log_every_n_steps": 1,
            "trainer.check_val_every_n_epoch": 1,
            "trainer.enable_checkpointing": True,
            "trainer.callbacks": callbacks,
            "data.epithets": default_epithets,
        })

        parser.link_arguments("data.task", "model.init_args.task", apply_on="instantiate")
        parser.link_arguments("data.base_transformer", "model.init_args.base_transformer", apply_on="instantiate")
        parser.link_arguments("data.model_class", "model.init_args.model_class", apply_on="instantiate")
        parser.link_arguments("data.num_labels", "model.init_args.num_labels", apply_on="instantiate")
        parser.link_arguments("data.tokenizer", "model.init_args.tokenizer", apply_on="instantiate")
        parser.link_arguments("data.batch_size", "model.init_args.batch_size", apply_on="instantiate")
        parser.link_arguments("data.id2label", "model.init_args.id2label", apply_on="instantiate")


def cli_main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    torch.set_float32_matmul_precision('medium')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cli = RhetoresGraecigCli(
        datamodule_class=AncientGreek,
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False
    )


if __name__ == "__main__":
    cli_main()