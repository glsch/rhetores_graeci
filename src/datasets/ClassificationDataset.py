import argparse
import os
import random
from typing import Literal, List, Union, Tuple, Dict

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
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.split = split

        self.df["label"], unique = pd.factorize(self.df["target"])

        self.id2label = {i: l for i, l in enumerate(unique)}
        self.label2id = {l: i for i, l in self.id2label.items()}

        self.sequences = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()

    def populate_authors(self):
        n = self.steps // self.window
        if self.steps % self.window != 0:
            n += 1
        expanded = self.orig_authors * n

        return expanded[:self.steps]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sampled_texts = []
        labels = []
        for _, i in enumerate(range(self.params.num_sample_per_author)):
            sampled_examples = self.sample_example_set(idx)
            sampled_texts.extend([i["text"] for i in sampled_examples])
            labels.extend([sampled_examples[0]["label"]])

        sampled_row = self.df[self.df["label"] == idx].sample(1).iloc[0]
        encoded_sample = self.tokenizer.encode_plus(
                sampled_row["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
                )
        encoded_sample = {k: v.reshape(self.params.num_sample_per_author, self.params.num_style_examples, -1) for k, v in encoded_sample.items()}
        encoded_sample["labels"] = torch.tensor(labels)

        return encoded_sample # each key is a tensor of shape (num_sample_per_author, num_style_examples, token_max_length), except for "labels" which is a tensor of shape (num_sample_per_author, num_style_examples)

if __name__ == "__main__":
    args = argparse.Namespace(tokenizer_model="nlpaueb/bert-base-greek-uncased-v1", num_style_examples=4, num_sample_per_author=4)
    df = pd.read_csv(os.path.join(PathManager.preprocessed_path, "rhetores_author.tsv"), sep="\t")
    df["label"], unique = pd.factorize(df["target"])
    # df.rename(columns={"target": "label"}, inplace=True)
    print(df)
    print(unique)
    preprocessed_dataset = ClassificationDataset(args, df)

    batch = []
    for i in range(4):
        print(type(preprocessed_dataset[i]))
        batch.append(preprocessed_dataset[i])


    new_batch = {}
    for k in batch[0].keys():
        new_batch[k] = torch.stack([i[k] for i in batch])

    labels = new_batch["labels"]
    print(labels)
    print(labels.shape)
    labels = labels.flatten()
    print(labels)
    print(labels.shape)




# class SupervisedAttributionDataset(Dataset):
#     """
#     Dataset class for authorship attribution. For each author (index of the dataset),
#     it samples N times a set of writing style examples. Each set of writing style examples consists of M documents.
#     Each document is a chunk of text of size `section_length` in specified units (tokens, words).
#     N is specified by `num_sample_per_author`, M is specified by `num_style_examples`
#     doing it  times.
#     Each set of writing style examples is tokenized using the tokenizer of the `base_transformer`.
#     Tokenization can be made on the set of writing style examples "as is" or with masking of x% of the content tokens by PoS.
#     Additionally, a x% of Byte-Pair Encodings can be masked.
#     The dataset can also extract manual stylometric features from the text, such as PoS ngrams,
#     MF words and lemmata, typed char ngrams, etc. However, this is not used at the moment.
#
#     Depending on the mode, the dataset can be used for supervised or self-supervised learning.
#     The dataset always returns a dictionary with keys: "data_df", "author", and, if features enabled, "feature_vector".
#     In supervised case, this dict remains intact.
#     In self-supervised case, the collate_fn checks whether the num_sample_per_author an even number >= 2 and splits the data_df in the dict to collate a batch
#     dataset returns a dictionary with keys: "anchor", "replica", "author", and, if features enabled, "anchor_fv" and "replica_fv". This is done by slicing
#     half of samples per author (this is why there should be, at least 2 samples per author) and stacking them in the batch.
#     """
#
#     def __init__(
#             self,
#             data: pd.DataFrame,
#             params: argparse.Namespace = None,
#             split: Literal["train", "dev", "test"] = "train"
#     ):
#         self.params = params
#         self.split = split
#         self.data = data[data["split"] == self.split]
#         self.data["target"] = self.data["label"].astype(int)
#
#         self.unique_targets = self.data["target"].unique().tolist()
#
#         self.targets = self.extend_dataset()
#
#         self.tokenizer = AutoTokenizer.from_pretrained(self.params.tokenizer_model)
#
#     def extend_dataset(self):
#         """
#         This method artificially extends the size of the dataset to ensure that each epoch can cycle through the dataset
#         multiple times, potentially helping the model to see every sample. This is particularly useful in scenarios where
#         the dataset has imbalanced classes of different sizes.
#
#         :return: A list of indices representing the extended dataset.
#         """
#         # "exploding" dataset length
#         total_steps = len(self.unique_targets) * self.params.explode_factor
#         if total_steps % self.params.batch_size != 0:
#             total_steps += self.params.batch_size - (total_steps % self.params.batch_size)
#
#         # minimum number of repetitions needed to reach at least 'total_steps' length
#         n_times = (total_steps + len(self.unique_targets) - 1) // len(self.unique_targets)  # Ceiling division
#
#         # extend the dataset by repeating the indices of unique authors
#         exploded = self.unique_targets * n_times
#
#         return exploded[:total_steps]
#
#     def sample_style_examples(self, idx: int) -> Dict[str, Union[list, Union[int, float]]]:  # -> List[Dict[str, any]]:
#         """
#         For datapoint 'idx':
#          1) Retrieves 'target'
#          2) Takes `num_style_examples` - 1 samples samples where 'target' is the same as 'target' of 'idx'
#         where 'target' is the same as  of writing style examples,
#         returning a list of str
#
#         :param idx: Index of the datapoint in self.data_df
#         :return:
#         """
#         # todo: reconsider this logic as it seems to overfit to the targets with many samples (maybe downsampling)
#         # 'target' at the index
#
#         target = self.targets[idx]
#
#         # all datapoints with the same 'target' as the index
#         data_subset = self.data[self.data["target"] == target].reset_index(drop=True)
#         num_docs = data_subset.shape[0]
#
#         # calculate the number of examples to return
#         num_style_examples = self.params.num_style_examples
#         style_example_set_size = min(num_docs, num_style_examples)
#
#         # we want to take 'num_style_examples' - 1 samples starting from a random index in the subset
#         maxval = num_docs - style_example_set_size
#         start_index = random.randint(0, maxval)
#
#         # indices are consecutive, but may be we should sample them less systematically
#         selected_indices = list(range(start_index, start_index + style_example_set_size))
#
#         data_subset = data_subset.iloc[selected_indices]["text"].tolist()
#
#         # style_example_set = core_sample + [{"text": doc, "target": target} for doc in data_subset]
#         style_example_set = data_subset
#
#         return {"text": style_example_set, "target": target}
#
#     def sample_random_window(self, data, window_length=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Samples a random window from the text.
#         Following the example, instead of learning from actual sentences, we take a group of sentences and learn from
#         random windows picked from each sentence in the group. As authors claim, this help with generalization and prevents overfit.
#         """
#
#         if window_length is None:
#             logger.info("Using default window length of 32 tokens, since no window length was provided.")
#             window_length = 32
#
#         input_ids, attention_mask = data
#
#         cls = self.tokenizer.cls_token_id
#         pad = self.tokenizer.pad_token_id
#         eos = self.tokenizer.eos_token_id
#         if type(eos) != int:
#             eos = self.tokenizer.sep_token_id
#
#         # Inputs are smaller than window size -> add padding
#         padding = window_length - input_ids.shape[1]
#         if padding > 0:
#             input_ids = F.pad(input_ids, (0, padding), 'constant', pad)
#             attention_mask = F.pad(attention_mask, (0, padding), 'constant', 0)
#             return input_ids, attention_mask
#
#         # Inputs are larger than window size -> sample random windows
#         true_lengths = torch.sum(torch.where(input_ids != 1, 1, 0), 1)
#
#         start_indices = torch.tensor(
#             [random.randint(1, l - window_length + 2) if l >= window_length else 1 for l in true_lengths])
#         indices = torch.tensor(
#             [list(range(start, start + window_length - 2)) for start, l in zip(start_indices, true_lengths)])
#         input_ids = input_ids.gather(1, indices)
#         attention_mask = attention_mask.gather(1, indices)
#
#         # Add cls token
#         input_ids = F.pad(input_ids, (1, 0), 'constant', cls)
#         attention_mask = F.pad(attention_mask, (1, 0), 'constant', 1)
#
#         # Add eos token
#         input_ids = torch.cat((input_ids, torch.where(true_lengths >= window_length, eos, pad).unsqueeze(1)), 1)
#         attention_mask = torch.cat((attention_mask, torch.where(true_lengths >= window_length, 1, 0).unsqueeze(1)), 1)
#
#         return input_ids, attention_mask
#
#     def mask_data_bpe(self, data):  # -> List[torch.Tensor]:
#         """Masks x% of Byte-Pair Encodings from the input.
#         """
#         if self.params.mask_bpe_percentage > 0.0:
#             mask = torch.rand(data["input_ids"].size()) >= (1. - self.params.mask_bpe_percentage)
#             pad_mask = ~(data["input_ids"] == self.tokenizer.pad_token_id)
#             # for k, v in data.items():
#             # mask = torch.rand(data[0].size()) >= (1. - self.params.mask_bpe_percentage)
#
#             # This is why we won't quite get to the mask percentage asked for by the user.
#             # pad_mask = ~(data[0] == self.tokenizer.pad_token_id)
#             mask *= pad_mask
#             # for k, v in data.items():
#             #     v.masked_fill_(mask, self.tokenizer.mask_token_id)
#             # data[0].masked_fill_(mask, self.tokenizer.mask_token_id)
#             return {k: v.masked_fill_(mask, self.tokenizer.mask_token_id) for k, v in data.items() if
#                     k in ["input_ids", "attention_mask"]}
#
#         else:
#             return data
#         # return data
#
#     def tokenize_text(self, example_set: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Tokenizes the author writing style example set.
#         :param example_set: List[str] of author writing style example set.
#         :return: List[torch.Tensor, torch.Tensor] with input_ids, attention_mask
#         of size (num_style_examples * num_sample_per_author, token_max_length)
#         """
#
#         is_window = False
#         if self.params.random_window is not None:
#             if self.params.random_window > self.tokenizer.model_max_length:
#                 raise ValueError("`random_window` cannot be longer than the maximum allowed input length "
#                                  "of the model associated to the tokenizer's (`model_max_length`)")
#             is_window = True
#             if self.params.random_window > 0:
#                 self.params.token_max_length = self.params.random_window
#
#         tokenized_example_set = self.tokenizer(
#             example_set,
#             padding=True if is_window else "max_length",
#             truncation=False if is_window else True,
#             max_length=None if is_window else self.params.token_max_length,
#             return_tensors='pt'
#         )
#
#         # tokenized_example_set = (tokenized_example_set.input_ids, tokenized_example_set.attention_mask)
#
#         return tokenized_example_set
#
#     def __len__(self):
#         return len(self.targets)
#
#     def __getitem__(self, idx: int):  # -> Dict[
#         # str, Union[Tuple[List[torch.Tensor]], None, List[str], torch.Tensor, str]]:
#         text = []
#         target = []
#         for _ in range(self.params.num_sample_per_author):
#             style_example_set = self.sample_style_examples(idx)
#             target.append(style_example_set["target"])
#             text.extend(style_example_set)
#
#         # size (num_sample_per_author)
#         # target = torch.tensor([d["target"] for d in text])
#         target = torch.tensor(target)
#
#         try:
#             assert target.dtype == torch.int64, "Labels must be integers"
#         except AssertionError:
#             target = target.to(torch.int64)
#         #
#
#         # text_to_tokenize = [doc["text"] for doc in text]
#         text_to_tokenize = text
#
#         data = self.tokenize_text(text_to_tokenize)
#
#         if self.params.random_window is not None and self.params.random_window > 0:
#             data = self.sample_random_window(data, window_length=self.params.random_window)
#
#         # reshaping input_ids and attention mask tensors to (num_sample_per_author, num_style_examples, token_max_length)
#         # data_df is a tuple: input_ids, attention_mask
#         # data = [d.reshape(self.params.num_sample_per_author, -1, self.params.token_max_length) for d in data]
#         data = {k: v.reshape(self.params.num_sample_per_author, -1, self.params.token_max_length) for k, v in
#                 data.items()}
#         target = target.reshape(self.params.num_sample_per_author, -1)
#
#         # print(target)
#         #
#
#         #
#         # print("before masking", data_df)
#
#         data = self.mask_data_bpe(data)
#         data["labels"] = target
#
#         return data

# class SupervisedAttributionDataset(Dataset):
#     """
#     Dataset class for authorship attribution. For each author (index of the dataset),
#     it samples N times a set of writing style examples. Each set of writing style examples consists of M documents.
#     Each document is a chunk of text of size `section_length` in specified units (tokens, words).
#     N is specified by `num_sample_per_author`, M is specified by `num_style_examples`
#     doing it  times.
#     Each set of writing style examples is tokenized using the tokenizer of the `base_transformer`.
#     Tokenization can be made on the set of writing style examples "as is" or with masking of x% of the content tokens by PoS.
#     Additionally, a x% of Byte-Pair Encodings can be masked.
#     The dataset can also extract manual stylometric features from the text, such as PoS ngrams,
#     MF words and lemmata, typed char ngrams, etc. However, this is not used at the moment.
#
#     Depending on the mode, the dataset can be used for supervised or self-supervised learning.
#     The dataset always returns a dictionary with keys: "data_df", "author", and, if features enabled, "feature_vector".
#     In supervised case, this dict remains intact.
#     In self-supervised case, the collate_fn checks whether the num_sample_per_author an even number >= 2 and splits the data_df in the dict to collate a batch
#     dataset returns a dictionary with keys: "anchor", "replica", "author", and, if features enabled, "anchor_fv" and "replica_fv". This is done by slicing
#     half of samples per author (this is why there should be, at least 2 samples per author) and stacking them in the batch.
#     """
#
#     def __init__(
#             self,
#             data: pd.DataFrame,
#             params: argparse.Namespace = None,
#             split: Literal["train", "dev", "test"] = "train"
#     ):
#         self.params = params
#         self.split = split
#         self.data = data[data["split"] == self.split]
#         self.data["target"] = self.data["label"].astype(int)
#
#
#
#         self.tokenizer = AutoTokenizer.from_pretrained(self.params.tokenizer_model)
#
#     def sample_style_examples(self, idx: int) -> Dict[str, Union[list, Union[int, float]]]: # -> List[Dict[str, any]]:
#         """
#         For datapoint 'idx':
#          1) Retrieves 'target'
#          2) Takes `num_style_examples` - 1 samples samples where 'target' is the same as 'target' of 'idx'
#         where 'target' is the same as  of writing style examples,
#         returning a list of str
#
#         :param idx: Index of the datapoint in self.data_df
#         :return:
#         """
#         # todo: reconsider this logic as it seems to overfit to the targets with many samples (maybe downsampling)
#         # 'target' at the index
#         target = self.data.iloc[idx]["target"]
#
#         # datapoint at the index
#         # core_sample = [{"text": self.data.iloc[idx]["text"], "target": target}]
#         core_sample = [self.data.iloc[idx]["text"]]
#
#         if self.params.num_style_examples == 1:
#             return {"text": core_sample, "target": target}
#
#         # all datapoints with the same 'target' as the index
#         data_subset = self.data[self.data["target"] == target].reset_index(drop=True)
#         num_docs = data_subset.shape[0]
#
#         # calculate the number of examples to return
#         num_style_examples = self.params.num_style_examples - 1
#         style_example_set_size = min(num_docs, num_style_examples)
#
#         # we want to take 'num_style_examples' - 1 samples starting from a random index in the subset
#         maxval = num_docs - style_example_set_size
#         start_index = random.randint(0, maxval)
#
#         # indices are consecutive, but may be we should sample them less systematically
#         selected_indices = list(range(start_index, start_index + style_example_set_size))
#
#         data_subset = data_subset.iloc[selected_indices]["text"].tolist()
#
#         # style_example_set = core_sample + [{"text": doc, "target": target} for doc in data_subset]
#         style_example_set = core_sample + data_subset
#
#         return {"text": style_example_set, "target": target}
#
#     def sample_random_window(self, data, window_length=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Samples a random window from the text.
#         Following the example, instead of learning from actual sentences, we take a group of sentences and learn from
#         random windows picked from each sentence in the group. As authors claim, this help with generalization and prevents overfit.
#         """
#
#         if window_length is None:
#             logger.info("Using default window length of 32 tokens, since no window length was provided.")
#             window_length = 32
#
#         input_ids, attention_mask = data
#
#         cls = self.tokenizer.cls_token_id
#         pad = self.tokenizer.pad_token_id
#         eos = self.tokenizer.eos_token_id
#         if type(eos) != int:
#             eos = self.tokenizer.sep_token_id
#
#         # Inputs are smaller than window size -> add padding
#         padding = window_length - input_ids.shape[1]
#         if padding > 0:
#             input_ids = F.pad(input_ids, (0, padding), 'constant', pad)
#             attention_mask = F.pad(attention_mask, (0, padding), 'constant', 0)
#             return input_ids, attention_mask
#
#         # Inputs are larger than window size -> sample random windows
#         true_lengths = torch.sum(torch.where(input_ids != 1, 1, 0), 1)
#
#         start_indices = torch.tensor(
#             [random.randint(1, l - window_length + 2) if l >= window_length else 1 for l in true_lengths])
#         indices = torch.tensor(
#             [list(range(start, start + window_length - 2)) for start, l in zip(start_indices, true_lengths)])
#         input_ids = input_ids.gather(1, indices)
#         attention_mask = attention_mask.gather(1, indices)
#
#         # Add cls token
#         input_ids = F.pad(input_ids, (1, 0), 'constant', cls)
#         attention_mask = F.pad(attention_mask, (1, 0), 'constant', 1)
#
#         # Add eos token
#         input_ids = torch.cat((input_ids, torch.where(true_lengths >= window_length, eos, pad).unsqueeze(1)), 1)
#         attention_mask = torch.cat((attention_mask, torch.where(true_lengths >= window_length, 1, 0).unsqueeze(1)), 1)
#
#         return input_ids, attention_mask
#
#     def mask_data_bpe(self, data): #  -> List[torch.Tensor]:
#         """Masks x% of Byte-Pair Encodings from the input.
#         """
#         if self.params.mask_bpe_percentage > 0.0:
#             mask = torch.rand(data["input_ids"].size()) >= (1. - self.params.mask_bpe_percentage)
#             pad_mask = ~(data["input_ids"] == self.tokenizer.pad_token_id)
#             #for k, v in data.items():
#             #mask = torch.rand(data[0].size()) >= (1. - self.params.mask_bpe_percentage)
#
#             # This is why we won't quite get to the mask percentage asked for by the user.
#             # pad_mask = ~(data[0] == self.tokenizer.pad_token_id)
#             mask *= pad_mask
#             # for k, v in data.items():
#             #     v.masked_fill_(mask, self.tokenizer.mask_token_id)
#             # data[0].masked_fill_(mask, self.tokenizer.mask_token_id)
#             return {k: v.masked_fill_(mask, self.tokenizer.mask_token_id) for k, v in data.items() if k in ["input_ids", "attention_mask"]}
#
#         else:
#             return data
#         # return data
#
#     def tokenize_text(self, example_set: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Tokenizes the author writing style example set.
#         :param example_set: List[str] of author writing style example set.
#         :return: List[torch.Tensor, torch.Tensor] with input_ids, attention_mask
#         of size (num_style_examples * num_sample_per_author, token_max_length)
#         """
#
#         is_window = False
#         if self.params.random_window is not None:
#             if self.params.random_window > self.tokenizer.model_max_length:
#                 raise ValueError("`random_window` cannot be longer than the maximum allowed input length "
#                                  "of the model associated to the tokenizer's (`model_max_length`)")
#             is_window = True
#             if self.params.random_window > 0:
#                 self.params.token_max_length = self.params.random_window
#
#         tokenized_example_set = self.tokenizer(
#             example_set,
#             padding=True if is_window else "max_length",
#             truncation=False if is_window else True,
#             max_length=None if is_window else self.params.token_max_length,
#             return_tensors='pt'
#         )
#
#         # tokenized_example_set = (tokenized_example_set.input_ids, tokenized_example_set.attention_mask)
#
#         return tokenized_example_set
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx: int): # -> Dict[
#         # str, Union[Tuple[List[torch.Tensor]], None, List[str], torch.Tensor, str]]:
#         text = []
#         target = []
#         for _ in range(self.params.num_sample_per_author):
#             style_example_set = self.sample_style_examples(idx)
#             target.append(style_example_set["target"])
#             text.extend(style_example_set)
#
#         # size (num_sample_per_author)
#         # target = torch.tensor([d["target"] for d in text])
#         target = torch.tensor(target)
#
#         try:
#             assert target.dtype == torch.int64, "Labels must be integers"
#         except AssertionError:
#             target = target.to(torch.int64)
#         #
#
#         # text_to_tokenize = [doc["text"] for doc in text]
#         text_to_tokenize = text
#
#         data = self.tokenize_text(text_to_tokenize)
#
#         if self.params.random_window is not None and self.params.random_window > 0:
#             data = self.sample_random_window(data, window_length=self.params.random_window)
#
#         # reshaping input_ids and attention mask tensors to (num_sample_per_author, num_style_examples, token_max_length)
#         # data_df is a tuple: input_ids, attention_mask
#         # data = [d.reshape(self.params.num_sample_per_author, -1, self.params.token_max_length) for d in data]
#         data = {k: v.reshape(self.params.num_sample_per_author, -1, self.params.token_max_length) for k, v in data.items()}
#         target = target.reshape(self.params.num_sample_per_author, -1)
#
#
#
#         # print(target)
#         #
#
#         #
#         # print("before masking", data_df)
#
#         data = self.mask_data_bpe(data)
#         data["labels"] = target
#
#         return data

        # print(target.shape)
        # print(data[0].shape)
        # print(data[1].shape)

        # item_dict = {
        #     "data": data, # shape (num_sample_per_author, num_style_examples, token_max_length)
        #     "target": target, # shape (num_sample_per_author, num_style_examples)
        #
        # }

        # return item_dict


