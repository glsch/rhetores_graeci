import ast
import enum
import os.path
from typing import List, Union

from tqdm import tqdm
tqdm.pandas()

from lightning.pytorch import LightningDataModule

from jsonargparse.typing import NonNegativeInt, NonNegativeFloat,ClosedUnitInterval, restricted_number_type

from src.datasets.PandasDataset import PandasDataset
from src.path_manager import PathManager
from src.datasets.utils import download_dataset
from transformers import AutoTokenizer, AutoModel, RobertaModel, DataCollatorForLanguageModeling, DefaultDataCollator

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.MlmTuningModule import AutoModelForMaskedLMWrapper
from src.classification.ClassificationModule import AutoModelForSequenceClassificationWrapper
from src.datasets.MlmDataset import MLMDataset
from src.datasets.ClassificationDataset import ClassificationDataset
from src.logger_config import logger

class TextChunkType(enum.Enum):
    SENTENCE = "sentence"
    CHUNK = "chunk"


class AncientGreekDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                 epithets: List[str]=None,
                 chunk_type: TextChunkType = TextChunkType.CHUNK,
                 batch_size: NonNegativeInt = 1,
                 num_workers: NonNegativeInt = 0,
                 persistent_workers: bool = False,
                 chunk_length: restricted_number_type("from_64_to_512", int, [(">=", 64), ("<=", 512)]) = 256,
                 overlap: restricted_number_type("from_00_to_09", float, [(">=", 0.0), ("<=", 0.9)]) = 0.5,
                 ):


        super().__init__()
        self.dataset_path = PathManager.dataset_path
        self.author_metadata_path = PathManager.author_metadata_path

        self.tokenizer = tokenizer
        self.chunk_type = chunk_type
        self.chunk_length = chunk_length
        self.overlap = overlap

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        if epithets is None:
            self.epithets = list()
        else:
            self.epithets = epithets

        assert isinstance(self.epithets, list), "Epithets must be a list"

        # self.fname = "preprocessed_dataset"
        # self.task = "mlm"
        #
        # if isinstance(self.trainer.model.model, AutoModelForMaskedLMWrapper):
        #     self.fname = "mlm_" + self.fname
        #     self.task = "mlm"
        # elif isinstance(self.trainer.model.model, AutoModelForSequenceClassificationWrapper):
        #     self.fname = "classification_" + self.fname
        #     self.task = "classification"

        self.dataset = None

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        def expand_levels(levels):
            row_dict = {}
            for i, (value, name) in enumerate(levels):
                row_dict[f'l{i}'] = value
                row_dict[f'l{i}_name'] = name
            return row_dict

        #if not os.path.exists(os.path.join(PathManager.data_path, "preprocessed", "preprocessed_dataset.csv")):
        if not (os.path.exists(self.dataset_path) and os.path.exists(self.author_metadata_path)):
            download_dataset()

        if not os.path.exists(os.path.join(PathManager.data_path, "preprocessed", "preprocessed_dataset.csv")):
            # opening dataset and metadata
            pd_dataset = PandasDataset(dataset_path=self.dataset_path, author_metadata_path=self.author_metadata_path)

            if len(self.epithets) > 0:
                filtered_authors = pd_dataset.select_authors_by_epithet(self.epithets)
                filtered_df = pd_dataset.df[pd_dataset.df["author_id"].isin(filtered_authors["author_id"])]
            else:
                filtered_df = pd_dataset.df

            # authors for the study
            study_author_ids = [284, 87, 607, 640, 594, 2002, 2178, 613, 1376, 592, 649, 560, 2586, 2903, 616, 605, 2027, 81]
            # authors which constituted UNK category in the article
            unk_author_ids = [607, 594, 2002, 2178, 649, 560, 186, 2903, 605]

            filtered_df["levels"] = filtered_df["levels"].fillna(method="ffill")

            dataset = filtered_df
            expanded_rows = []

            for i, r in tqdm(dataset.iterrows(), total=dataset.shape[0]):
                levels = ast.literal_eval(r["levels"])
                new_r = {}
                for k, v in r.to_dict().items():
                    if k == "levels":
                        continue
                    else:
                        if k in ("author_id", "work_id"):
                            new_r[k] = int(v)
                        else:
                            new_r[k] = str(v)

                new_r.update(expand_levels(levels))
                expanded_rows.append(new_r)

            dataset = pd.DataFrame(data=expanded_rows)

            grouped_dfs = []
            for i, ((author_id, author_label, work_id), df) in enumerate(dataset.groupby(["author_id", "author_label", "work_id"])):
                # todo: add hermogenean divisions here
                if author_id == 81 and work_id == 16:
                    for j, (chapter, chapter_df) in enumerate(df.groupby("l1")):
                        chapter_df = pd.DataFrame(
                            data=[{"text": " ".join(chapter_df["text"].tolist()), "author_id": author_id, "work_id": work_id, "siglum": f"{author_label}_AR_{chapter}", 'target': author_label}])

                        grouped_dfs.append(chapter_df)

                elif author_id in (592, 2586, 284, 2027, 87):
                    for w, (work_id, work_df) in enumerate(df.groupby("work_id")):
                        # for the future verification task
                        siglum = f"{author_label}_{work_id}_uncertain"
                        work_df = pd.DataFrame(
                            data=[{"text": " ".join(work_df["text"].tolist()), "author_id": author_id, "work_id": work_id, 'target': author_label, 'siglum': siglum}])

                        grouped_dfs.append(work_df)

                else:
                    new_df = pd.DataFrame(data=[
                        {"text": " ".join(df["text"].tolist()), "author_id": author_id, "work_id": work_id,
                         'target': author_label, 'siglum': f"{author_label}_{work_id}"}])

                    grouped_dfs.append(new_df)

            dataset = pd.concat(grouped_dfs)
            dataset = dataset.dropna(subset=['text'])

            all_chunks = []
            if self.chunk_type == TextChunkType.CHUNK:
                for i, r in tqdm(dataset.iterrows(), total=dataset.shape[0]):
                    tokens = self.tokenizer(r["text"], add_special_tokens=False).input_ids
                    row_chunks = [tokens[i:i + self.chunk_length] for i in range(0, len(tokens), self.chunk_length - int(self.chunk_length * self.overlap))]
                    all_chunks.append(row_chunks)

                chunks = [self.tokenizer.batch_decode(row_chunks) for row_chunks in all_chunks]

            elif self.chunk_type == TextChunkType.SENTENCE:
                for i, r in tqdm(dataset.iterrows(), total=dataset.shape[0]):
                    all_chunks.append(sent_tokenize(r["text"]))

                chunks = all_chunks

            else:
                raise ValueError("Invalid chunk type")

            dataset = dataset.assign(chunks=chunks)

            dataset = dataset.explode("chunks")
            dataset = dataset.dropna(subset=['chunks'])
            dataset = dataset.reset_index(drop=True)

            self.dataset = dataset

            self.dataset.drop(columns=["text"], inplace=True)

            self.dataset = self.dataset.assign(split="train")
            self.dataset.rename(columns={"chunks": "text"}, inplace=True)

            self.dataset = self.dataset.reset_index(drop=True)

            if self.task == "mlm":
                self.dataset = self.dataset.sample(frac=1.0)
                self.train = self.dataset.iloc[:int(0.8 * self.dataset.shape[0])]
                self.dataset.loc[self.train.index, "split"] = "train"
                self.val = self.dataset.iloc[int(0.8 * self.dataset.shape[0]):int(0.9 * self.dataset.shape[0])]
                self.dataset.loc[self.val.index, "split"] = "val"

                self.test = self.dataset.iloc[int(0.9 * self.dataset.shape[0]):]
                self.dataset.loc[self.test.index, "split"] = "test"

            elif self.task == "classification":
                logger.info(f"AncientGreekDataModule.prepare_data() -- Creating classification dataset")

                # retain only the authors for the study
                self.dataset = self.dataset[self.dataset["author_id"].isin(study_author_ids) | self.dataset["author_id"].isin(unk_author_ids)]
                # Dionysius Ars Rhetorica goes to predict corpus
                predict_df = self.dataset[self.dataset["author_id"] == 81 & self.dataset["work_id"] == 16]
                predict_df = predict_df.assign(split="predict")

                self.dataset = self.dataset[~(self.dataset["author_id"] == 81 & self.dataset["work_id"] == 16)]

                # all minor authors go to UNK, which will only be in the test set
                # therefore, we can create the base of this dataset
                unk_df = self.dataset[self.dataset["author_id"].isin(unk_author_ids)]
                unk_df = unk_df.assign(split="test")

                # the rest is used for training and validation
                self.dataset = self.dataset[~self.dataset["author_id"].isin(unk_author_ids)]

                train_df = self.dataset.groupby("author_id").sample(frac=.75)
                train_df = train_df.assign(split="train")

                val_df = self.dataset[~self.dataset.index.isin(train_df.index)]
                test_df = val_df.groupby("author_id").sample(frac=.5)
                val_df = val_df[~val_df.index.isin(test_df.index)]
                val_df = val_df.assign(split="val")
                test_df = test_df.assign(split="test")

                # todo: add label encoding somewhere here

                self.dataset = pd.concat([train_df, val_df, test_df, unk_df, predict_df])

            #self.dataset.to_csv(os.path.join(PathManager.data_path, "preprocessed", f"{self.fname}.csv"), index=False)
            self.dataset.to_csv(os.path.join(PathManager.data_path, "preprocessed", f"preprocessed_dataset.csv"), index=False)

        else:
            self.dataset = pd.read_csv(os.path.join(PathManager.data_path, "preprocessed", f"preprocessed_dataset.csv"))


    def setup(self, stage: str) -> None:
        dataset_cls = None
        self.collate_fn = None
        self.dataset = pd.read_csv(os.path.join(PathManager.data_path, "preprocessed", f"preprocessed_dataset.csv"))

        self.train_df = self.dataset[self.dataset["split"] == "train"]
        self.val_df = self.dataset[self.dataset["split"] == "val"]
        self.test_df = self.dataset[self.dataset["split"] == "test"]

        if isinstance(self.trainer.model.model, AutoModelForMaskedLMWrapper):
            logger.info(f"AncientGreekDataModule.setup() -- Model is subclass of {AutoModelForMaskedLMWrapper}: {self.trainer.model.model.__class__.__name__}")
            dataset_cls = MLMDataset
            self.collate_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        elif isinstance(self.trainer.model.model, AutoModelForSequenceClassificationWrapper):
            logger.info(
                f"AncientGreekDataModule.setup() -- Model is subclass of {AutoModelForMaskedLMWrapper}: {self.trainer.model.model.__class__.__name__}")
            dataset_cls = ClassificationDataset
            self.collate_fn = DefaultDataCollator(return_tensors="pt")

        else:
            logger.info(
                f"AncientGreekDataModule.setup() -- Model is {self.trainer.model.model} {self.trainer.model.model.__class__} {self.trainer.model.model.__class__.__name__} {self.trainer.model.model.__dir__}")
            raise ValueError("Invalid model")

        if stage == "fit":
            self.train_dataset = dataset_cls(df=self.train_df, split="train", tokenizer=self.tokenizer)
            self.val_dataset = dataset_cls(df=self.val_df, split="val", tokenizer=self.tokenizer)

        elif stage == "test":
            self.val_dataset = dataset_cls(df=self.val_df, split="val", tokenizer=self.tokenizer)
            self.test_dataset = dataset_cls(df=self.test_df, split="test", tokenizer=self.tokenizer)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    dm = AncientGreekDataModule(epithets=["Rhet.", "Orat."], tokenizer=tokenizer, chunk_type=TextChunkType.CHUNK, overlap=0.0, chunk_length=512)

    dm.prepare_data()