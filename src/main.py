import os

from pathlib import Path
from jsonargparse import lazy_instance
from lightning import Trainer, LightningModule
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.cli import SaveConfigCallback
import torch
import numpy as np
import random

from jsonargparse import class_from_function

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, Callable

from src.datasets.AncientGreekDataModule import AncientGreekDataModule as AncientGreek
from src.classification.ClassificationModule import ClassificationModule # noqa: F401
from src.MlmTuningModule import MlmTuningModule # noqa: F401
from src.callbacks import PushToHuggingfaceCallback
from lightning.fabric.utilities.cloud_io import _is_dir, _is_local_file_protocol, get_filesystem
from lightning.fabric.utilities.types import _PATH
from src.path_manager import PathManager

from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment

from typing import Optional


# class Model2(DemoModel):
#     def configure_optimizers(self):
#         print("⚡", "using Model2", "⚡")
#         return super().configure_optimizers()



class RhetoresGraecigCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        logger_config = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            'init_args': {
                "project": "RhetoresGraeci",
                "log_model": "all"
            },
        }

        default_classifier = {
            "class_path": "src.classification.ClassificationModule.AutoModelForSequenceClassificationWrapper",
            'init_args': {
                "pretrained_model_name_or_path": "bowphs/GreBerta"
            },
        }

        callbacks = [
            {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint"},
            {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
             "init_args": {"logging_interval": "step"}},
            {"class_path": "src.callbacks.PushToHuggingfaceCallback", "init_args": {"repo_owner": "glsch"}},
            {"class_path": "lightning.pytorch.callbacks.EarlyStopping", "init_args": {"min_delta": 0.01, "patience": 3, "monitor": "val/loss", "mode": "min", "check_on_train_epoch_end": False, "verbose": True}},
            # lazy_instance(SaveConfigCallback)
        ]

        default_epithets = [
            "Rhet.",
            "Orat."
        ]

        parser.set_defaults({
            # "model.transformer": default_classifier,
            "trainer.logger": logger_config,
            # "trainer.logger": lazy_instance(WandbLogger, project="PatristicStylometryClustering", log_model="all"),
            "trainer.precision": 32,
            "trainer.max_epochs": 30,
            "trainer.log_every_n_steps": 1,
            "trainer.check_val_every_n_epoch": 1,
            "trainer.enable_checkpointing": True,
            "trainer.callbacks": callbacks,
            "data.epithets": default_epithets,
            # # this does not work correctly as config would contain a 'proj_size' key which is only supported by LSTM, but is
            # # also present in the config for any subclass of RNNBase
            #"model.scheduler_type": {"class_path": "transformers.SchedulerType.COSINE"},
            # "model.postprocessing": {"class_path": "src.models.clustering.normalizers.Pipeline",
            #                          "init_args": {"steps": default_postprocessing}},
            # "model.noise": {"class_path": "src.models.clustering.RnnClusterer.GaussianNoise",
            #                 "init_args": {"sigma": 0.5, "is_relative_detach": True}},
            # "model.head": {"class_path": "src.models.clustering.RnnClusterer.OutputLayer", "init_args": {
            #     "softmax_layer": {"class_path": "torch.nn.LogSoftmax", "init_args": {"dim": 2}}},
            #                },

            # "model.lr_scheduler_rnn.init_args.start_factor": 1,
            # "model.lr_scheduler_rnn.init_args.end_factor": 0.0,
            # "model.lr_scheduler_rnn.total_iters": 1000,
            # "model.lr_scheduler_head.init_args.start_factor": 1,
            # "model.lr_scheduler_head.init_args.end_factor": 0.0,
            # "model.lr_scheduler_head.init_args.total_iters": 1000

        })

        parser.link_arguments("model.tokenizer", "data.tokenizer", apply_on="instantiate")
        parser.link_arguments("model.model", "data.model", apply_on="instantiate")
        # parser.link_arguments("model.tokenizer", "data.tokenizer", apply_on="instantiate")
        # parser.link_arguments("model.num_heads", "trainer.num_sanity_val_steps", apply_on="instantiate")
        #
        # parser.link_arguments("data.alphabet_len", "model.rnn.init_args.input_size", apply_on="instantiate")
        # parser.link_arguments("model.rnn.init_args.hidden_size", "model.head.init_args.input_dim",
        #                       apply_on="instantiate")
        # parser.link_arguments("data.alphabet_len", "model.head.init_args.output_dim", apply_on="instantiate")

        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR, "model.lr_scheduler_rnn", )
        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.LinearLR, "model.lr_scheduler_head")

        # parser.set_defaults(
        #
        # )

        # dynamic_class = class_from_function(instantiate_myclass)
        # parser.add_class_arguments(dynamic_class, "myclass.init")


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