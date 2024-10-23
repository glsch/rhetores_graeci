# python standard modules
from typing import Literal, Union, Any

# third-party modules
import pandas as pd
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# project modules
from src.logger_config import logger

class ClassificationDataset(Dataset):
    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            df: pd.DataFrame,
            split: Literal["train", "val", "test", "predict"] = "train",
            **kwargs: Any
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.split = split

        self.records = self.df.to_dict(orient="records")

        logger.debug(f"Loaded {len(self.records)} records for {self.split} split.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        tokenized = self.tokenizer(
            self.records[idx]["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        if self.split == "predict":
            tokenized["siglum"] = self.records[idx]["siglum"]

        tokenized["label"] = self.records[idx]["label"]

        logger.debug(f"Tokenized record {idx} for {self.split} split.")
        logger.debug(f"Tokenized: {tokenized}.")

        return tokenized

if __name__ == "__main__":
    pass