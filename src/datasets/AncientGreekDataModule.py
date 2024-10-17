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
from pytorch_metric_learning.samplers import MPerClassSampler

class TextChunkType(enum.Enum):
    SENTENCE = "sentence"
    CHUNK = "chunk"


class AncientGreekDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                 model: torch.nn.Module=None,
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
        self.task = "mlm"
        self.fname = "preprocessed_dataset"
        if isinstance(model, AutoModelForMaskedLMWrapper):
            logger.info(f"AncientGreekDataModule.__init__() -- Model is subclass of {AutoModelForMaskedLMWrapper}: {model.__class__.__name__}")

            self.fname = "mlm_" + self.fname
        elif isinstance(model, AutoModelForSequenceClassificationWrapper):
            logger.info(
                f"AncientGreekDataModule.__init__() -- Model is subclass of {AutoModelForSequenceClassificationWrapper}: {model.__class__.__name__}")
            self.task = "classification"
            self.fname = "classification_" + self.fname

        logger.info(f"AncientGreekDataModule.__init__() -- Task {self.task}")

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

        self.dataset = None
        self._num_classes = None

        self.prepared = False

    # @property
    # def num_classes(self):
    #     if self._num_classes is None:
    #         model = AutoModelForSequenceClassificationWrapper()
    #         self.setup(stage="fit")
    #
    #     return self._num_classes

    def prepare_data(self) -> None:
        logger.info(
            f"AncientGreekDataModule.prepare_data()")
        #if self.prepared:
        #    return

        def expand_levels(levels):
            row_dict = {}
            for i, (value, name) in enumerate(levels):
                row_dict[f'l{i}'] = value
                row_dict[f'l{i}_name'] = name
            return row_dict

        #if not os.path.exists(os.path.join(PathManager.data_path, "preprocessed", "preprocessed_dataset.csv")):
        if not (os.path.exists(self.dataset_path) and os.path.exists(self.author_metadata_path)):
            download_dataset()

        if not os.path.exists(os.path.join(PathManager.data_path, "preprocessed", f"{self.fname}.csv")):
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
                predict_df = self.dataset[(self.dataset["author_id"] == 81) & (self.dataset["work_id"] == 16)]
                predict_df = predict_df.assign(split="predict")

                self.dataset = self.dataset[~((self.dataset["author_id"] == 81) & (self.dataset["work_id"] == 16))]

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

                logger.info("AncientGreekDataModule.prepare_data() -- Creating dataset")
                self.dataset = pd.concat([train_df, val_df, test_df])
                encoded_labels, unique = pd.factorize(self.dataset["author_id"])
                max_label = max(encoded_labels)
                unk = max_label + 1
                self.dataset = self.dataset.assign(label=encoded_labels)
                unk_df = unk_df.assign(target="<UNK>")
                unk_df = unk_df.assign(label=unk)
                predict_df = predict_df.assign(label=unk + 1)
                self.dataset = pd.concat([self.dataset, unk_df, predict_df])
                logger.info(f"AncientGreekDataModule.prepare_data() -- Number of authors full dataset: {self.dataset['author_id'].unique().tolist()}")
                logger.info(f"AncientGreekDataModule.prepare_data() -- Number of authors train df: {train_df['author_id'].unique().tolist()}")
                logger.info(f"AncientGreekDataModule.prepare_data() -- Number of authors unk_df: {unk_df['author_id'].unique().tolist()}")
                logger.info(f"AncientGreekDataModule.prepare_data() -- Number of authors val_df: {val_df['author_id'].unique().tolist()}")

                logger.info(f"AncientGreekDataModule.prepare_data() -- Max samples: {self.dataset[self.dataset['split'].isin(['train', 'val'])].groupby('author_id').size().max()}")
                logger.info(f"AncientGreekDataModule.prepare_data() -- Min samples: {self.dataset[self.dataset['split'].isin(['train', 'val'])].groupby('author_id').size().min()}")
                logger.info(f"AncientGreekDataModule.prepare_data() -- Counts: {self.dataset[self.dataset['split'].isin(['train', 'val'])].groupby('author_id').size().sort_values()}")


                logger.info(f"AncientGreekDataModule.prepare_data() -- Dataset columns: {self.dataset.columns}")

                # todo: add label encoding somewhere here
            self.dataset.to_csv(os.path.join(PathManager.data_path, "preprocessed", f"{self.fname}.csv"), index=False)
            #self.prepared = True
        else:
            self.dataset = pd.read_csv(os.path.join(PathManager.data_path, "preprocessed", f"{self.fname}.csv"))

        self.id2label = self.dataset[self.dataset["split"].isin(["train", "val"])][["label", "target"]].drop_duplicates().set_index("label")["target"].to_dict()

    def setup(self, stage: str, model: torch.nn.Module = None) -> None:
        logger.info(
            f"AncientGreekDataModule.setup()")
        dataset_cls = None
        self.collate_fn = None

        # if model is not None and isinstance(model, AutoModelForSequenceClassificationWrapper):
        #     self.task = "classification"
        #     if not os.path.exists(os.path.join(PathManager.data_path, "preprocessed", f"{self.fname}.csv")):
        #         self.prepare_data()

        if model is None:
            model = self.trainer.model.model

        self.dataset = pd.read_csv(os.path.join(PathManager.data_path, "preprocessed", f"{self.fname}.csv"))

        self.train_df = self.dataset[self.dataset["split"] == "train"]
        self.val_df = self.dataset[self.dataset["split"] == "val"]
        self.test_df = self.dataset[self.dataset["split"] == "test"]

        self.sampler = None

        if isinstance(model, AutoModelForMaskedLMWrapper):
            logger.info(f"AncientGreekDataModule.setup() -- Model is subclass of {AutoModelForMaskedLMWrapper}: {self.trainer.model.model.__class__.__name__}")
            dataset_cls = MLMDataset
            self.collate_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        elif isinstance(model, AutoModelForSequenceClassificationWrapper):
            self.id2label = self.dataset[self.dataset["split"].isin(["train", "val"])][["label", "target"]].drop_duplicates().set_index("label")["target"].to_dict()
            self.label2id = {l: i for i, l in self.id2label.items()}
            self._num_classes = len(self.id2label)
            self.trainer.model.num_classes = self._num_classes
            logger.info(
                f"AncientGreekDataModule.setup() -- Model is subclass of {AutoModelForMaskedLMWrapper}: {self.trainer.model.model.__class__.__name__}")
            dataset_cls = ClassificationDataset
            self.collate_fn = DefaultDataCollator(return_tensors="pt")

            m = (self.batch_size // self._num_classes) + 1
            self.sampler = [
                MPerClassSampler(self.train_df["label"], m=m,
                                            length_before_new_iter=len(self.train_df) * 100),
                MPerClassSampler(self.val_df["label"], m=m,
                                            length_before_new_iter=len(self.val_df) * 100),
                MPerClassSampler(self.test_df["label"], m=m,
                                 length_before_new_iter=len(self.test_df) * 100),
            ]


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
        sampler = None
        if self.sampler is not None:
            if isinstance(self.sampler, list):
                sampler = self.sampler[0]

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=True if sampler is None else False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        return loader

    def val_dataloader(self):
        sampler = None
        if self.sampler is not None:
            if isinstance(self.sampler, list):
                sampler = self.sampler[1]

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        return loader

    def test_dataloader(self):
        sampler = None
        if self.sampler is not None:
            if isinstance(self.sampler, list):
                sampler = self.sampler[2]

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        return loader

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    model = AutoModelForSequenceClassificationWrapper(pretrained_model_name_or_path="bowphs/GreBerta")
    dm = AncientGreekDataModule(epithets=["Rhet.", "Orat."], tokenizer=tokenizer, model=model, chunk_type=TextChunkType.CHUNK, overlap=0.5, chunk_length=128)

    dm.prepare_data()