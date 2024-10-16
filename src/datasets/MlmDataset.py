import argparse
from functools import partial
import os.path
import random
from typing import Literal, List, Union, Tuple, Dict

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel, RobertaModel, DataCollatorForLanguageModeling

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.path_manager import PathManager
from src.logger_config import logger

from src.datasets.PandasDataset import PandasDataset

from transformers import PreTrainedTokenizer, AutoTokenizer

tqdm.pandas()

class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM) task.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            df: pd.DataFrame,
            split: Literal["train", "dev", "test"] = "train",
    ):

        self.split = split
        self.tokenizer = tokenizer
        logger.info(f"MLMDataset.__init__() -- Using tokenizer with vocab_size: {self.tokenizer.vocab_size}")

        self.sequences = df["chunks"].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        tokenized = self.tokenizer(
            self.sequences[idx],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True
        )

        return tokenized

if __name__ == "__main__":
    from src.datasets.AncientGreekDataModule import AncientGreekDataModule, TextChunkType
    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    dm = AncientGreekDataModule(epithets=["Rhet.", "Orat."], tokenizer=tokenizer, chunk_type=TextChunkType.SENTENCE)

    dm.prepare_data()

    from transformers import RobertaForMaskedLM

    model = RobertaForMaskedLM.from_pretrained("bowphs/GreBerta")

    mlm_dataset = MLMDataset(df=dm.dataset, split="train", tokenizer=tokenizer)
    # print(attr_dataset[0])

    batch = []
    for i in range(10):
        #print(mlm_dataset[i].input_ids.shape)
        # print(bert(**mlm_dataset[i]))
        it = mlm_dataset[i]
        print(it)
        batch.append(it)

    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    train_loader = DataLoader(
        mlm_dataset,
        batch_size=4,
        collate_fn=collate_fn
    )

    #batch = collate_fn(batch)

    for _, b in enumerate(train_loader):
        if _ == 4:
            break

        print(model(**b))