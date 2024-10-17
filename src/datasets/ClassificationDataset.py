import argparse
import os
import random
from typing import Literal, List, Union, Tuple, Dict, Any

import pandas as pd

import torch
from transformers import AutoTokenizer, BatchEncoding
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer


from src.logger_config import logger
from src.path_manager import PathManager

class ClassificationDataset(Dataset):
    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            df: pd.DataFrame,
            split: Literal["train", "val", "test"] = "train",
            **kwargs: Any
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.split = split

        # self.df["label"], unique = pd.factorize(self.df["target"])

        #self.id2label = {i: l for i, l in enumerate(unique)}
        #self.label2id = {l: i for i, l in self.id2label.items()}

        self.records = self.df.to_dict(orient="records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        tokenized = self.tokenizer(
            self.records[idx]["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        tokenized["label"] = self.records[idx]["label"]

        return tokenized

if __name__ == "__main__":
    from src.datasets.AncientGreekDataModule import AncientGreekDataModule, TextChunkType
    from transformers import DefaultDataCollator
    from torch.utils.data import DataLoader
    from pytorch_metric_learning.samplers import MPerClassSampler

    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    dm = AncientGreekDataModule(epithets=["Rhet.", "Orat."], tokenizer=tokenizer, chunk_type=TextChunkType.SENTENCE)

    dm.prepare_data()

    from transformers import RobertaForSequenceClassification

    model = RobertaForSequenceClassification.from_pretrained("bowphs/GreBerta")

    mlm_dataset = ClassificationDataset(df=dm.dataset, split="train", tokenizer=tokenizer)

    batch = []
    for i in range(10):
        # print(mlm_dataset[i].input_ids.shape)
        # print(bert(**mlm_dataset[i]))
        it = mlm_dataset[i]
        # print(it)
        batch.append(it)

    collate_fn = DefaultDataCollator(return_tensors="pt")

    train_loader = DataLoader(
        mlm_dataset,
        batch_size=32,
        sampler=MPerClassSampler(mlm_dataset.labels, m=3, length_before_new_iter=10000),
        collate_fn=collate_fn
    )

    # batch = collate_fn(batch)
    for _, b in enumerate(train_loader):
        if _ == 4:
            break

        print(b)


    print(len(mlm_dataset.labels))
    print(mlm_dataset.id2label)
    print(len(mlm_dataset.df.siglum.unique().tolist()))

        # print(model(**b))

