# python standard modules
from typing import Literal, Union, Any

# third-party modules
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
tqdm.pandas()
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer
)

# project modules
from src.logger_config import logger

class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM) task.
    """

    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            df: pd.DataFrame,
            split: Literal["train", "val", "test"] = "train",
            **kwargs: Any
    ):

        self.split = split
        self.tokenizer = tokenizer
        logger.info(f"MLMDataset.__init__() -- Using tokenizer with vocab_size: {self.tokenizer.vocab_size}")

        self.sequences = df["text"].tolist()

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
    pass