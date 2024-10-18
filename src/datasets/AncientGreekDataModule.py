import ast
import enum
import os.path
from typing import List, Union, Type
from typing import ForwardRef

import transformers.models.auto
from tqdm import tqdm
tqdm.pandas()

from lightning.pytorch import LightningDataModule

from jsonargparse.typing import NonNegativeInt, NonNegativeFloat,ClosedUnitInterval, restricted_number_type

from src.datasets.PandasDataset import PandasDataset
from src.path_manager import PathManager
from src.datasets.utils import download_dataset
from transformers import AutoTokenizer, AutoModel, RobertaModel, DataCollatorForLanguageModeling, DefaultDataCollator, DataCollatorWithPadding, AutoModelForMaskedLM, AutoModelForSequenceClassification

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.datasets.MlmDataset import MLMDataset
from src.datasets.ClassificationDataset import ClassificationDataset
from src.logger_config import logger
from pytorch_metric_learning.samplers import MPerClassSampler
from dataclasses import dataclass

class TextChunkType(enum.Enum):
    SENTENCE = "sentence"
    CHUNK = "chunk"

def resolve_forward_ref(ref: Union[str, ForwardRef, Type]):
    if isinstance(ref, str):
        ref = ForwardRef(ref)
    if isinstance(ref, ForwardRef):
        return ref._evaluate(globals(), locals(), set())
    return ref


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    siglum: str = None

    def __call__(self, features):
        # Call the parent class to handle the default collation
        batch = super().__call__(features)

        # Handle the additional key
        if self.siglum in features[0]:
            batch[self.siglum] = [f[self.siglum] for f in features]

        return batch


class AncientGreekDataModule(LightningDataModule):
    def __init__(self,
                 model_class: Union[Type[AutoModelForMaskedLM], Type[AutoModelForSequenceClassification]] = transformers.models.auto.AutoModelForSequenceClassification,
                 base_transformer: str = "bowphs/GreBerta",
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
        self.base_transformer = base_transformer
        self.model_class = model_class

        self.fname = "preprocessed_dataset"
        logger.info(f"AncientGreekDataModule.__init__() -- Model class: {self.model_class}, {type(self.model_class)} Type: {Type[self.model_class]}")
        logger.info(f"AncientGreekDataModule.__init__() -- Target model class: {Type[AutoModelForMaskedLM]}")
        if self.model_class == AutoModelForMaskedLM:
            self.task = "mlm"
        elif self.model_class == AutoModelForSequenceClassification:
            self.task = "classification"
        else:
            raise ValueError(f"Invalid model class! Expected 'AutoModelForSequenceClassification' or 'AutoModelForMaskedLM', got {self.model_class}")

        self.fname = f"{self.task}_{self.fname}.csv"
        self.preprocessed_dataset_path = os.path.join(PathManager.data_path, "preprocessed", self.fname)

        logger.info(f"AncientGreekDataModule.__init__() -- Task {self.task}, corresponding file: {self.preprocessed_dataset_path}")

        if self.base_transformer == "altsoph/bert-base-ancientgreek-uncased":
            self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_transformer)

        self.chunk_type = chunk_type
        self.chunk_length = chunk_length
        self.overlap = overlap

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        # authors for the study
        self.study_author_ids = [284, 87, 607, 640, 594, 2002, 2178, 613, 1376, 592, 649, 560, 2586, 2903, 616, 605, 2027,
                            81]
        # authors which constituted UNK category in the article
        self.unk_author_ids = [607, 594, 2002, 2178, 649, 560, 186, 2903, 605]

        if epithets is None:
            self.epithets = list()
        else:
            self.epithets = epithets

        assert isinstance(self.epithets, list), "Epithets must be a list"

        self.dataset = None
        self._num_labels = None
        self._id2label = None
        self._label2id = None

        self.save_hyperparameters()

    @property
    def num_labels(self):
        if self._num_labels is None:
            if self.task == "classification":
                if not os.path.exists(self.preprocessed_dataset_path):
                    self.prepare_data()

                self.setup(stage="fit")

        return self._num_labels

    @property
    def id2label(self):
        if self._id2label is None:
            if self.task == "classification":
                if not os.path.exists(self.preprocessed_dataset_path):
                    self.prepare_data()

                self.setup(stage="fit")

        return self._id2label

    @property
    def label2id(self):
        if self._label2id is None:
            if self.task == "classification":
                if not os.path.exists(self.preprocessed_dataset_path):
                    self.prepare_data()

                self.setup(stage="fit")

        return self._label2id

    def prepare_data(self) -> None:
        logger.info(f"AncientGreekDataModule.prepare_data()")
        def expand_levels(levels):
            row_dict = {}
            for i, (value, name) in enumerate(levels):
                row_dict[f'l{i}'] = value
                row_dict[f'l{i}_name'] = name
            return row_dict

        if not (os.path.exists(self.dataset_path) and os.path.exists(self.author_metadata_path)):
            download_dataset()

        if not os.path.exists(self.preprocessed_dataset_path):
            # opening dataset and metadata
            pd_dataset = PandasDataset(dataset_path=self.dataset_path, author_metadata_path=self.author_metadata_path)

            if len(self.epithets) > 0:
                filtered_authors = pd_dataset.select_authors_by_epithet(self.epithets)
                filtered_df = pd_dataset.df[pd_dataset.df["author_id"].isin(filtered_authors["author_id"])]
            else:
                filtered_df = pd_dataset.df

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

            self.dataset = self.dataset.reset_index(drop=False)
            self.dataset.rename(columns={"index": "unique_id"}, inplace=True)

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
                self.dataset = self.dataset[self.dataset["author_id"].isin(self.study_author_ids) | self.dataset["author_id"].isin(self.unk_author_ids)]
                # Dionysius Ars Rhetorica goes to predict corpus
                predict_df = self.dataset[(self.dataset["author_id"] == 81) & (self.dataset["work_id"] == 16)]
                predict_df = predict_df.assign(split="predict")
                predict_df = predict_df.assign(siglum=lambda x: x["siglum"].str.split("_").str[-1].astype(int))

                self.dataset = self.dataset[~self.dataset["unique_id"].isin(predict_df["unique_id"])]

                # all minor authors go to UNK, which will only be in the test set
                # therefore, we can create the base of this dataset
                unk_df = self.dataset[self.dataset["author_id"].isin(self.unk_author_ids)]
                unk_df = unk_df.assign(split="test")

                # the rest is used for training and validation
                self.dataset = self.dataset[~self.dataset["author_id"].isin(self.unk_author_ids)]

                train_df = self.dataset.groupby("author_id").sample(frac=.75)
                train_df = train_df.assign(split="train")

                val_df = self.dataset[~self.dataset["unique_id"].isin(train_df["unique_id"])]
                test_df = val_df.groupby("author_id").sample(frac=.5)
                val_df = val_df[~val_df["unique_id"].isin(test_df["unique_id"])]
                val_df = val_df.assign(split="val")
                test_df = test_df.assign(split="test")

                logger.info("AncientGreekDataModule.prepare_data() -- Creating dataset")
                self.dataset = pd.concat([train_df, val_df, test_df])
                encoded_labels, unique = pd.factorize(self.dataset["author_id"])
                max_label = max(encoded_labels)
                unk = max_label + 1
                self.dataset = self.dataset.assign(label=encoded_labels)
                unk_df = unk_df.assign(target="<UNK>")
                unk_df = unk_df.assign(label=-100)
                predict_df = predict_df.assign(label=-100)
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
            self.dataset.to_csv(self.preprocessed_dataset_path, index=False)
        else:
            self.dataset = pd.read_csv(self.preprocessed_dataset_path)

        if self.task == "classification":
            self._id2label = self.dataset[self.dataset["split"] != "predict"][["label", "target"]].drop_duplicates().set_index("label")["target"].to_dict()

    def setup(self, stage: str) -> None:
        logger.info(f"AncientGreekDataModule.setup() -- Stage: {stage}")
        dataset_cls = None
        self.collate_fn = None
        self.sampler = None

        if self.dataset is None:
            self.dataset = pd.read_csv(self.preprocessed_dataset_path)

        else:
            logger.info(f"AncientGreekDataModule.setup() -- Dataset already loaded")

        self.train_df = self.dataset[self.dataset["split"] == "train"]
        self.val_df = self.dataset[self.dataset["split"] == "val"]
        self.test_df = self.dataset[self.dataset["split"] == "test"]
        self.predict_df = self.dataset[self.dataset["split"] == "predict"]

        self.predict_df = self.predict_df.assign(siglum=lambda x: x["siglum"].astype(int))

        if self.task == "mlm":
            prohibited_ids = [i for i in self.study_author_ids + self.unk_author_ids if i != 81]

            self.train_df = self.train_df[~self.train_df["author_id"].isin(prohibited_ids)]
            self.val_df = self.val_df[~self.val_df["author_id"].isin(prohibited_ids)]
            self.test_df = self.test_df[~self.test_df["author_id"].isin(prohibited_ids)]
            dataset_cls = MLMDataset
            self.collate_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        elif self.task == "classification":
            # getting labels
            self._id2label = self.dataset[self.dataset["split"] != "predict"][["label", "target"]].drop_duplicates().set_index("label")["target"].to_dict()
            del self._id2label[-100]
            self._label2id = {l: i for i, l in self._id2label.items()}
            self._num_labels = len(self._id2label)
            dataset_cls = ClassificationDataset

            self.collate_fn = DataCollatorWithPadding(return_tensors="pt", tokenizer=self.tokenizer, padding="max_length", max_length=512)

            m = (self.batch_size // self._num_labels) + 1
            # length_before_new_iter = int(100 * self.batch_size)
            # length_before_new_iter = self.train_df.shape[0] * 5
            # defining MPerClassSampler's for DataLoaders
            self.sampler = [
                MPerClassSampler(self.train_df["label"].tolist(), m=m, length_before_new_iter=self.train_df.shape[0]),
                MPerClassSampler(self.val_df["label"].tolist(), m=m, length_before_new_iter=self.val_df.shape[0]),
                MPerClassSampler(self.test_df["label"].tolist(), m=m, length_before_new_iter=self.test_df.shape[0]),
            ]

        else:
            raise ValueError("Invalid model task and model type")

        if stage == "fit":
            self.train_dataset = dataset_cls(df=self.train_df, split="train", tokenizer=self.tokenizer)
            self.val_dataset = dataset_cls(df=self.val_df, split="val", tokenizer=self.tokenizer)
            self.test_dataset = dataset_cls(df=self.test_df, split="test", tokenizer=self.tokenizer)

        elif stage == "test":
            self.val_dataset = dataset_cls(df=self.val_df, split="val", tokenizer=self.tokenizer)
            self.test_dataset = dataset_cls(df=self.test_df, split="test", tokenizer=self.tokenizer)

        elif stage == "predict":
            self.val_dataset = dataset_cls(df=self.val_df, split="val", tokenizer=self.tokenizer)
            self.predict_dataset = dataset_cls(df=self.predict_df, split="predict", tokenizer=self.tokenizer)

            print(self.predict_df["siglum"].unique().tolist())


    def get_sampler(self, stage="train"):
        stage2idx = {"train": 0, "val": 1, "test": 2}
        assert stage in stage2idx, f"Invalid stage: {stage}"
        sampler = None
        if self.sampler is not None and isinstance(self.sampler, list):
            sampler = self.sampler[stage2idx[stage]]

        return sampler

    def train_dataloader(self):
        sampler = self.get_sampler(stage="train")
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=self.get_sampler(),
            shuffle=True if sampler is None else False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        return loader

    def val_dataloader(self):
        sampler = self.get_sampler(stage="val")
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
        sampler = self.get_sampler(stage="test")
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

    def predict_dataloader(self):
        sampler = None
        loader = DataLoader(
            self.predict_dataset,
            batch_size=1,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
        return loader

if __name__ == "__main__":
    #tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta")
    #model = AutoModelForSequenceClassificationWrapper(pretrained_model_name_or_path="bowphs/GreBerta")
    #dm = AncientGreekDataModule(epithets=["Rhet.", "Orat."], tokenizer=tokenizer, model=model, chunk_type=TextChunkType.CHUNK, overlap=0.5, chunk_length=128)
    dm = AncientGreekDataModule(epithets=["Rhet.", "Orat."], model_class=AutoModelForSequenceClassification, chunk_type=TextChunkType.CHUNK, overlap=0.5, chunk_length=128)

    # dm.prepare_data()