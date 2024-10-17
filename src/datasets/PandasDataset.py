from argparse import Namespace
from functools import partial
import os
import re
from typing import List, Union, Tuple, Any, Dict
import unicodedata

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize


from src.path_manager import PathManager
# from ..utils import download_preprocessed_corpus
from src.logger_config import logger
import ast

tqdm.pandas()

class PandasDataset:
    def __init__(
            self,
            dataset_path: str = PathManager.dataset_path,
            author_metadata_path: str = PathManager.author_metadata_path
                 ):
        self.dataset_path = dataset_path
        self.author_metadata_path = author_metadata_path

        self._author_metadata_df = None
        self.df = None

        self.df = pd.read_csv(self.dataset_path, dtype={"author_id": int})

        self.df = pd.merge(self.df, self.author_metadata_df, on="author_id", how="left")
        self.df = self.df.assign(author_label=self.df["author"].astype(str).str.split().str.join('_').str.lower() + "_" + self.df["author_id"].astype(str))
        self.df = self.normalize_to_ascii(self.df, "author_label")
        self.df = self.df.assign(text=self.df["text"].apply(lambda x: x.replace("‐ \n", "")))
        self.df = self.df.assign(text=self.df["text"].apply(lambda x: " ".join(x.split("\n"))))

    def _preprocess_text(self, df, column: str):
        pass

    def normalize_to_ascii(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        def _to_ascii(text):
            # Normalize Unicode characters to their closest ASCII equivalent
            normalized = unicodedata.normalize('NFKD', str(" ".join(text.split("\n"))))
            # Remove non-ASCII characters
            ascii_text = re.sub(r'[^\x00-\x7F]+', '', normalized)
            return ascii_text

        # Apply the normalization function to the specified column
        df[column] = df[column].apply(_to_ascii)

        return df

    def _load_author_metadata(self):
        logger.info(f"PandasDataset._load_author_metadata() -- Getting loading order...")
        self._author_metadata_df = pd.read_csv(self.author_metadata_path, dtype={"author_id": int})
        # additional processing for the author metadata
        self._author_metadata_df["epithet"] = self._author_metadata_df["epithet"].apply(
            lambda x: x if pd.notna(x) else "Unknown")
        # self._author_metadata_df = self._author_metadata_df.assign(
        #     epithet=lambda x: x["epithet"].apply(
        #         lambda y: y.split("\n") if isinstance(y, str) and "\n" in y else [y]
        #     )
        # )
        # self._author_metadata_df = self._author_metadata_df.explode("epithet", ignore_index=True)
        # self._author_metadata_df = self._author_metadata_df.assign(epithet=self._author_metadata_df["epithet"].str.replace("〈", "").str.replace("〉", "").str.strip())
        self._author_metadata_df["region"] = self._author_metadata_df["region"].apply(
            lambda x: x if pd.notna(x) else "Unknown")
        self._author_metadata_df["period"] = self._author_metadata_df["period"].apply(
            lambda x: x if pd.notna(x) else "Unknown")
        self._author_metadata_df.drop_duplicates(inplace=True)

        if "author_name" in self._author_metadata_df.columns:
            self._author_metadata_df.rename(columns={"author_name": "author"}, inplace=True)

    @property
    def author_metadata_df(self):
        if self._author_metadata_df is None:
            logger.info(f"PandasDataset.author_metadata_df() -- Loading author metadata...")
            self._load_author_metadata()
        return self._author_metadata_df

    def select_authors_by_epithet(self, epithets: List[str]):
        def contains_epithet_generator(df, column, values):
            for value in values:
                yield df[column].str.contains(value)

        # authors for the study
        study_author_ids = [284, 87, 607, 640, 594, 2002, 2178, 613, 1376, 592, 649, 560, 2586, 2903, 616, 605,
                            2027, 81]
        # authors which constituted UNK category in the article
        unk_author_ids = [607, 594, 2002, 2178, 649, 560, 186, 2903, 605]

        expression = pd.concat(contains_epithet_generator(self.author_metadata_df, "epithet", epithets), axis=1).any(
            axis=1)
        filtered_by_epithet = self.author_metadata_df[
            self.author_metadata_df["epithet"].notna()
            & (expression
            # adding DIONYSIUS HALICARNASSENSIS, too
               | (self.author_metadata_df["author_id"].isin(study_author_ids)) | self.author_metadata_df["author_id"].isin(unk_author_ids))][
            ["epithet", "author_id", "author", "period"]].sort_values(by="author", ascending=True)

        return filtered_by_epithet






# class PandasDataset:
#     def __init__(
#             self,
#             params: Namespace = None
#                  ):
#         """
#         Initializes the PandasDataset object, which is used in all subsequent pytorch Datasets.
#         This object is responsible for loading or building of the desired dataset, and for splitting it into
#         training, development, and test sets.
#
#         PandasDataset ensures that the minimal necessary amount of data is loaded needed to build the desired dataset.
#         Unless specified, the dataset is built from the core dataset, which is the full dataset.
#         By default, all task-related and study-related datasets are saved, which is why, in order to rebuild them from scratch
#         (to change splitting strategy or filtering, for example), it is important to delete the corresponding files.
#         -- mlm_dataset.csv or classification_dataset.csv
#         -- rhet_mlm_dataset.csv or rhet_classification_dataset.csv (or any other study)
#
#         At the end, it must have the following columns:
#         "text" for text data_df, and "target" for labels, whathever they are.
#         :param params:
#         """
#
#         self.params = params
#         self.random_state = self.params.random_seed # for splitting in train, dev, test
#
#         # initializing core dataframes
#         self._author_metadata_df = None # dataframe for default metadata
#         if not os.path.exists(PathManager.author_metadata_path):
#             logger.info(f"PandasDataset.author_metadata_df() -- Downloading archive with author metadata...")
#             download_preprocessed_corpus()
#         self.data_df = None # df which will be preprocessed
#         self._core_data_df = None # full dataset, will be loaded if needed to build subsets
#         self._task_data_df = None # mlm or classification data: to save time on preprocessing
#         self._study_data_df = None # task dataset: rhet and future ones
#         self.target = self.params.target # target, if any
#         self.id2label = None
#
#         # what kind of dataset we are interested in (henceforth, big dataset)
#         self.task = "mlm" if self.params.mode == "mlm" else "classification"
#
#         if self.task == "classification" and self.target is None:
#             raise ValueError("PandasDataset -- Target column must be specified for classification task.")
#
#         # filenames for different datasets
#         self.core_data_df_path = PathManager.dataset_path
#         self.task_data_df_path = os.path.join(PathManager.preprocessed_path, f"{self.params.mode}_dataset.csv")
#         self.study_data_df_path = os.path.join(PathManager.preprocessed_path, f"{self.params.study}_{self.params.mode}_dataset.csv")
#
#         logger.info(f"PandasDataset -- Core data path: {self.core_data_df_path}, exists {os.path.exists(self.core_data_df_path)}")
#         logger.info(f"PandasDataset -- Core data path: {self.task_data_df_path}, exists  {os.path.exists(self.task_data_df_path)}")
#         logger.info(f"PandasDataset -- Core data path: {self.study_data_df_path}, exists  {os.path.exists(self.study_data_df_path)}")
#
#         # hierarchy to decide what should be loaded to create the target df, if needed
#         self.dataset_hierarchy = {
#                 self.core_data_df_path: None,
#                 self.task_data_df_path: self.core_data_df_path,
#                 self.study_data_df_path: self.task_data_df_path
#         }
#
#         self.datasets_to_load = []
#
#         # initializing splits
#         self._train_df: pd.DataFrame = None
#         self._dev_df: pd.DataFrame = None
#         self._test_df: pd.DataFrame = None
#
#         # first we check whether the user provided a custom dataset
#         # its possible to pass custom dataset path as long as it has the necessary columns
#         # df must contain "author_id", "author_name", "work_id", possible "target" columns, and "text"
#         if self.params.input is not None:
#             self.data_df = self._load_custom_dataset(self.params.input)
#             if self.data_df is None:
#                 logger.info(f"PandasDataset -- Custom dataset was not found. Exiting.")
#                 exit(1)
#         else:
#             # otherwise, we check whether a small task-related dataset was requested
#             if self.params.study is not None:
#                 # we check whether the study dataset exists
#                 self.data_df = self.study_data_df
#             else:
#                 # whether task dataset exists
#                 self.data_df = self.task_data_df
#
#         assert "split" in self.data_df.columns, "Column 'split' not found in the dataset."
#
#
#     @property
#     def author_metadata_df(self):
#         if self._author_metadata_df is None:
#             logger.info(f"PandasDataset.author_metadata_df() -- Loading author metadata...")
#             self._load_author_metadata()
#         return self._author_metadata_df
#
#     @property
#     def core_data_df(self):
#         if self._core_data_df is None:
#             if not os.path.exists(PathManager.dataset_path):
#                 logger.info(f"PandasDataset.author_metadata_df() -- Downloading core dataset...")
#                 download_preprocessed_corpus()
#             self._core_data_df = pd.read_csv(PathManager.dataset_path, dtype={"author_id": int, "work_id": int})
#             self._core_data_df["text"].fillna("", inplace=True)
#             self._core_data_df["levels"] = self._core_data_df["levels"].fillna(method="ffill")
#
#         return self._core_data_df
#
#     @property
#     def task_data_df(self):
#         if self._task_data_df is None:
#             logger.info(f"PandasDataset.author_metadata_df() -- Building task dataset...")
#             if not os.path.exists(self.task_data_df_path):
#                 self._build_task_dataset(save=True)
#             else:
#                 self._task_data_df = pd.read_csv(self.task_data_df_path, dtype={"author_id": int, "work_id": int})
#                 if self.task == "classification":
#                     self.id2label = self.get_id2label(self._task_data_df)
#         return self._task_data_df
#
#     def get_id2label(self, df):
#         if not "label" in df.columns:
#             raise ValueError("Column 'label' not found in the dataset.")
#
#         unique = df["label"].unique()
#         id2label = {}
#
#         for id in unique:
#             label = df[df["label"] == id]["target"].iloc[0]
#             id2label[id] = label
#
#         return id2label
#
#     @property
#     def study_data_df(self):
#         if self._study_data_df is None:
#             logger.info(f"PandasDataset.author_metadata_df() -- Building study dataset...")
#             if not os.path.exists(self.study_data_df_path):
#                 self._build_study_dataset(save=True)
#             else:
#                 self._study_data_df = pd.read_csv(self.study_data_df_path, dtype={"author_id": int, "work_id": int})
#                 if self.task == "classification":
#                     if self.task == "classification":
#                         self.id2label = self.get_id2label(self._study_data_df)
#         return self._study_data_df
#
#
#     def _load_author_metadata(self):
#         logger.info(f"PandasDataset._load_author_metadata() -- Getting loading order...")
#         if not os.path.exists(PathManager.author_metadata_path):
#             logger.info(f"PandasDataset.author_metadata_df() -- Downloading core dataset...")
#             download_preprocessed_corpus()
#
#         self._author_metadata_df = pd.read_csv(PathManager.author_metadata_path, dtype={"author_id": int})
#         # additional processing for the author metadata
#         self._author_metadata_df["epithet"] = self._author_metadata_df["epithet"].apply(
#             lambda x: x if pd.notna(x) else "Unknown")
#         self._author_metadata_df["region"] = self._author_metadata_df["region"].apply(
#             lambda x: x if pd.notna(x) else "Unknown")
#         self._author_metadata_df["period"] = self._author_metadata_df["period"].apply(
#             lambda x: x if pd.notna(x) else "Unknown")
#         self._author_metadata_df.drop_duplicates(inplace=True)
#
#         if "author_name" in self._author_metadata_df.columns:
#             self._author_metadata_df.rename(columns={"author_name": "author"}, inplace=True)
#
#     def _build_task_dataset(self, save=False):
#         logger.info(f"PandasDataset._build_task_dataset() -- Building task dataset...")
#         if self.task == "mlm":
#             # grouping by author and work and joining the text: they will be split later, if needed
#             self._task_data_df = self.core_data_df.groupby(["author_id", "work_id"], as_index=False).agg(
#                 {"text": " ".join, "author_id": "first", "work_id": "first"})
#
#             self._task_data_df = self._preprocess_data(
#                 self._task_data_df,
#                 group=True,
#                 grouping_cols=["author_id", "work_id"],
#                 agg_actions={"author_id": "first", "work_id": "first", "text": " ".join },
#                 split=self.params.split,
#                 chunk_length=self.params.chunk_length,
#                 unit=self.params.unit
#             )
#
#             logger.info(f"PandasDataset._build_study_dataset() -- Splitting study MLM...")
#             self._task_data_df = self._preprocess_data(
#                 self._task_data_df,
#                 split="chunk",
#                 chunk_length=512,
#                 unit="token"
#             )
#
#             self._task_data_df = self._split_data(self.task_data_df)
#
#             if save:
#                 self._task_data_df.to_csv(self.task_data_df_path, index=False)
#
#             return self._task_data_df
#
#         else:
#             logger.info(f"PandasDataset._build_task_dataset() -- Building classification task dataset...")
#             # if we prepare for classification, we need to expand levels to allow elaborate grouping
#             def expand_levels(levels):
#                 row_dict = {}
#                 for i, (value, name) in enumerate(levels):
#                     row_dict[f'l{i}'] = value
#                     row_dict[f'l{i}_name'] = name
#                 return row_dict
#
#             logger.info(f"PandasDataset._build_task_dataset() -- Expanding levels: {self.core_data_df.shape}")
#             expanded_rows = []
#             for i, r in tqdm(self.core_data_df.iterrows(), total=self.core_data_df.shape[0]):
#                 levels = ast.literal_eval(r["levels"])
#                 new_r = {}
#                 for k, v in r.to_dict().items():
#                     if k == "levels":
#                         continue
#                     else:
#                         if k in ("author_id", "work_id"):
#                             new_r[k] = int(v)
#                         else:
#                             new_r[k] = str(v)
#
#                 new_r.update(expand_levels(levels))
#                 expanded_rows.append(new_r)
#
#             task_df = pd.DataFrame(data=expanded_rows)
#             logger.info(f"PandasDataset._build_task_dataset() -- Levels expanded...")
#             logger.info(f"PandasDataset._build_task_dataset() -- Task dataframe {task_df.shape}")
#
#             logger.info(f"PandasDataset._build_task_dataset() -- Columns in {self.task} dataset: {task_df.columns}...")
#             self._task_data_df = pd.merge(task_df, self.author_metadata_df, on="author_id", how="left")
#             self._task_data_df.rename(columns={self.target: "target"}, inplace=True)
#
#             if save:
#                 self._task_data_df.to_csv(self.task_data_df_path, index=False)
#
#             return self._task_data_df
#
#     def _build_study_dataset(self, save=False):
#         logger.info(f"PandasDataset._build_study_dataset() -- Building study dataset...")
#
#         self.selected_authors_mlm = self.author_metadata_df[
#             self.author_metadata_df["epithet"].notna()
#             & ((self.author_metadata_df["epithet"].str.contains("Rhet.")
#                 | self.author_metadata_df["epithet"].str.contains("Orat.")
#                 )
#                | (self.author_metadata_df["author_id"] == 81))][
#             ["epithet", "author_id", "author", "period"]].sort_values(by="author", ascending=True)
#
#         logger.info(f"PandasDataset._build_study_dataset() -- Selected {self.selected_authors_mlm['author_id'].unique().tolist()} ({len(self.selected_authors_mlm['author_id'].unique().tolist())}) rhetoricians and orators.")
#
#         if self.task == "mlm":
#             logger.info(f"PandasDataset._build_study_dataset() -- Filtering for MLM...")
#             logger.info(f"PandasDataset._build_study_dataset() -- Shape before filtering: {self.task_data_df.shape}")
#             self._study_data_df = self.task_data_df[self.task_data_df["author_id"].isin(self.selected_authors_mlm["author_id"])]
#             logger.info(f"PandasDataset._build_study_dataset() -- Shape before filtering: {self._study_data_df.shape}")
#
#             if save:
#                 self._study_data_df.to_csv(self.study_data_df_path, index=False)
#
#             return self._study_data_df
#
#         else:
#             logger.info(f"PandasDataset._build_study_dataset() -- Filtering for Sequence Classification...")
#             logger.info(f"PandasDataset._build_study_dataset() -- Shape before filtering: {self.task_data_df.shape}")
#             self.selected_authors_attribution = self.selected_authors_mlm[
#                 self.selected_authors_mlm["period"].str.contains(r"A\.\s*D\.") | (self.selected_authors_mlm["author_id"] == 81)]
#
#             logger.info(f"PandasDataset._build_study_dataset() -- Selected {self.selected_authors_attribution['author_id'].unique().tolist()} ({len(self.selected_authors_attribution['author_id'].unique().tolist())}) authors AD for {self.params.study}...")
#
#
#             study_df = self.task_data_df[self.task_data_df["author_id"].isin(self.selected_authors_attribution["author_id"])]
#             logger.info(f"PandasDataset._build_study_dataset() -- Shape after filtering: {study_df.shape}")
#             grouped_dfs = []
#             for i, ((author_id, work_id), df) in enumerate(study_df.groupby(["author_id", "work_id"])):
#                 # todo: add hermogenean divisions here
#                 if author_id == 81 and work_id == 16:
#                     for j, (chapter, chapter_df) in enumerate(df.groupby("l1")):
#                         chapter_df = pd.DataFrame(
#                             data=[{"text": " ".join(chapter_df["text"].tolist()), "author_id": author_id, "work_id": work_id, "siglum": f"AR_{chapter}", 'target': chapter_df["target"].iloc[0]}])
#
#                         grouped_dfs.append(chapter_df)
#
#                 elif author_id in (592, 2586, 284, 2027, 87):
#                     for w, (work_id, work_df) in enumerate(df.groupby("work_id")):
#                         if author_id == 592:
#                             siglum = f"HERM_{work_id}"
#                         elif author_id == 2586:
#                             siglum = f"MEN_{work_id}"
#                         elif author_id == 87:
#                             siglum = f"AE_HEROD_{work_id}"
#                         elif author_id == 284:
#                             siglum = f"AE_ARIST_{work_id}"
#                         elif author_id == 2027:
#                             siglum = f"VAL_APS_{work_id}"
#                         else:
#                             siglum = f"UNKNOWN_{work_id}"
#
#                         work_df = pd.DataFrame(
#                             data=[{"text": " ".join(work_df["text"].tolist()), "author_id": author_id, "work_id": work_id, 'target': work_df["target"].iloc[0], 'siglum': siglum}])
#
#                         grouped_dfs.append(work_df)
#
#                 else:
#                     new_df = pd.DataFrame(data=[
#                         {"text": " ".join(df["text"].tolist()), "author_id": author_id, "work_id": work_id,
#                          'target': df["target"].iloc[0], 'siglum': work_id}])
#
#                     grouped_dfs.append(new_df)
#
#             self._study_data_df = pd.concat(grouped_dfs)
#
#             logger.info(f"PandasDataset._build_study_dataset() -- Shape after filtering: {study_df.shape}")
#
#             logger.info(f"PandasDataset._build_study_dataset() -- Study dataset {self.params.study} created with shape {self._study_data_df.shape}")
#             logger.info(f"PandasDataset._build_study_dataset() -- Columns {self._study_data_df.columns}...")
#             logger.info(f"PandasDataset._build_study_dataset() -- Authors {self._study_data_df['author_id'].unique().tolist()}...")
#
#             self._study_data_df = self._preprocess_data(
#                 self._study_data_df,
#                 group=True,
#                 grouping_cols=["author_id", "work_id", "siglum"],
#                 agg_actions={"author_id": "first", "work_id": "first", "text": " ".join, "siglum": "first", "target": "first"},
#                 split="sentence",
#                 unit=self.params.unit,
#                 chunk_length=self.params.chunk_length
#             )
#             self._study_data_df = self._split_data(self.study_data_df)
#             self._study_data_df = pd.merge(self._study_data_df, self.author_metadata_df, on="author_id", how="left")
#
#             def filter_authors(row):
#                 authors_to_retain = [3094, 2200, 284, 87, 607, 640, 594, 2002, 2178, 613, 81, 1376, 592, 649, 560, 186, 2586, 2903, 616, 605, 2027]
#                 if row["author_id"] == 81 and row["work_id"] == 16:
#                     return "test"
#                 else:
#                     return row["split"]
#
#             # self._study_data_df = self._study_data_df.apply(filter_authors, axis=1)
#
#             if save:
#                 self._study_data_df.to_csv(self.study_data_df_path, index=False)
#
#             return self._study_data_df
#
#     def _load_custom_dataset(self, path: str):
#         """
#             Loads a custom dataset from the given path.
#             :param path:
#             :return:
#             """
#         # if custom dataset does not exist, we return None
#         if not os.path.exists(path):
#             logger.error(f"PandasDataset._load_custom_dataset() -- Dataset does not exist. Returning {None}.")
#             return None
#         # otherwise, we check that the dataset has the necessary columns
#         else:
#             logger.info(f"PandasDataset._load_custom_dataset() -- Custom dataset found. Loading from {path}")
#             data_df = pd.read_csv(path, dtype={"author_id": int, "work_id": int})
#             assert "author_id" in data_df.columns, "Column 'author_id' not found in the dataset."
#             assert "work_id" in data_df.columns, "Column 'work_id' not found in the dataset."
#
#             if self.task == "classification":
#                 if self.target is None:
#                     raise ValueError("PandasDataset -- Target column must be specified for classification task.")
#
#                 assert self.target in data_df.columns, f"Target column '{self.target}' not found in the dataset."
#
#         return data_df
#
#     @property
#     def train_df(self) -> pd.DataFrame:
#         """Returns the training data_df as a Pandas dataframe."""
#         if self._train_df is None:
#             self._train_df = self.data_df[self.data_df["split"] == "train"]
#         return self._train_df
#
#     @property
#     def dev_df(self) -> pd.DataFrame:
#         """Returns the training data_df as a Pandas dataframe."""
#         if self._dev_df is None:
#             self._dev_df = self.data_df[self.data_df["split"] == "dev"]
#         return self._dev_df
#
#     @property
#     def test_df(self) -> pd.DataFrame:
#         """Returns the training data_df as a Pandas dataframe."""
#         if self._test_df is None:
#             self._test_df = self.data_df[self.data_df["split"] == "test"]
#
#         if self._test_df.shape[0] == 0:
#             # logger.info(f"PandasDataset.test_df() -- Test data_df is empty. Returning dev data_df.")
#             return self._dev_df
#         else:
#             return self._test_df
#
#     def _preprocess_data(self, df, group=False, preprocess=True, grouping_cols: List[str]=None, agg_actions: Dict[str,Any]=None, split="sentence", chunk_length=512, unit="word"):
#         logger.info(f"PandasDataset._preprocess_data() -- Preprocessing dataset")
#         logger.info(f"PandasDataset._preprocess_data() -- Columns: {df.columns}...")
#
#         if group:
#             if grouping_cols is None:
#                 grouping_cols = ["author_id", "work_id"]
#
#             if agg_actions is None:
#                 agg_actions = {"author_id": "first", "work_id": "first", "text": " ".join}
#
#             # grouping
#             logger.info(f"PandasDataset._preprocess_data() -- Grouping by {grouping_cols} with action {agg_actions}")
#             df = df.groupby(grouping_cols).agg(agg_actions)
#             logger.info(f"PandasDataset._preprocess_data() -- Cleaning...")
#
#             df["text"] = df["text"].progress_apply(
#                     lambda x: re.sub(r"(\w+)‐\s*(\w+)", repl=r"\1\2", string=x, flags=re.DOTALL))
#             logger.info(f"PandasDataset._preprocess_data() -- Splitting into sentences...")
#             df["text"] = df["text"].progress_apply(sent_tokenize)
#             df = df.explode("text").reset_index(drop=True)
#             df["text"].fillna("", inplace=True)
#             df = df[~(df["text"] == "")]
#             logger.info(f"PandasDataset._preprocess_data() -- Done...")
#
#         if preprocess:
#             if split == "sentence":
#                 logger.info(f"PandasDataset._preprocess_data() -- {split}: no actions needed")
#                 pass
#             else:
#                 logger.info(f"PandasDataset._preprocess_data() -- Splitting into {split} of {chunk_length} {unit}")
#
#                 if unit == "word":
#                     def split_into_chunks_of_words(text, chunk_length):
#                         chunks = []
#                         words = word_tokenize(text)
#                         for i in range(0, len(words), chunk_length):
#                             chunk_units = words[i:i + chunk_length]
#                             chunk = " ".join(chunk_units)
#                             chunks.append(chunk)
#
#                         return chunks
#
#                     split_fn = partial(split_into_chunks_of_words, chunk_length=chunk_length)
#
#                 elif unit == "token":
#                     tokenizer = AutoTokenizer.from_pretrained(self.params.tokenizer_model)
#                     def split_into_chunks_of_tokens(text, tokenizer, chunk_length):
#                         tokens = tokenizer(text, add_special_tokens=False).input_ids
#                         chunked = [tokens[chunk: chunk + chunk_length] for chunk in
#                                    range(0, len(tokens), chunk_length)]
#                         chunked = [tokenizer.batch_decode([c])[0] for c in chunked if len(c) > 0]
#
#                         return chunked
#
#                     split_fn = partial(split_into_chunks_of_tokens, tokenizer=tokenizer, chunk_length=chunk_length)
#
#                 else:
#                     raise ValueError(f"PandasDataset._preprocess_data() -- Unit {unit} not recognized.")
#
#                 df["text"] = df["text"].progress_apply(split_fn)
#                 df = df.explode("text").reset_index(drop=True)
#
#
#                 logger.info(
#                     f"PandasDataset._preprocess_data() -- Done")
#
#             return df
#
#     def _split_data(self, df):
#         """
#         Splits the data_df into training, development, and test sets.
#         :return:
#         """
#         logger.info(f"PandasDataset._split_data() -- Splitting data_df into train, dev, test...")
#         if self.task == "classification":
#             data_points_per_target = df["target"].value_counts()
#             median_per_target = data_points_per_target.median()
#             # other_threshold = median_per_target
#
#             joined_df = pd.merge(data_points_per_target, self.author_metadata_df, left_on='target', right_on='author', how='left')
#
#             for i, ((author_id, author_name), d_df) in enumerate(joined_df.groupby(["author_id", "author"])):
#                 logger.info(f"PandasDataset._split_data() -- ({author_id}): {author_name} -- {d_df['period'].iloc[0]} -- {d_df['region'].iloc[0]}")
#
#             logger.info(f"PandasDataset._split_data() -- Data points per target: {data_points_per_target}, median: {median_per_target}...")
#
#             # todo: consider other threshold criteria here
#             other_threshold = self.params.other_below
#
#             # todo: consider author disjoint, too
#             # those which have less than average data_df points get "other" category
#             df["target"] = df["target"].apply(lambda x: x if pd.notna(x) and data_points_per_target[x] >= other_threshold else "Other")
#             df["target"] = df["target"].apply(lambda x: x if pd.notna(x) and not ("MARCELLINI SCHOLIA AD HERMOGENIS STATUS" in x) else "Other")
#             trains, devs, tests = [], [], []
#
#             targets = df["target"].unique().tolist()
#
#             logger.info(f"PandasDataset._split_data() -- Targets {targets} ({len(targets)})...")
#
#             # AR goes to test
#             for (author_id, target), target_df in df.groupby(["author_id", "target"]):
#                 logger.info(f"PandasDataset._split_data() -- Splitting target {target}")
#                 if author_id == 81:
#                     ar = target_df[target_df["work_id"] == 16]
#                     target_df = target_df[~(target_df.index.isin(ar.index))]
#                     train_df = target_df.sample(frac=0.7, random_state=self.random_state).reset_index(drop=True)
#                     dev_df = target_df[~(target_df.index.isin(train_df.index))]
#                     trains.append(train_df)
#                     devs.append(dev_df)
#                     tests.append(ar)
#                     continue
#
#                 train_df = target_df.sample(frac=0.7, random_state=self.random_state).reset_index(drop=True)
#                 dev_df = target_df[~target_df.index.isin(train_df.index)]
#                 test_df = dev_df.sample(frac=0.5, random_state=self.random_state).reset_index(drop=True)
#                 dev_df = dev_df[~dev_df.index.isin(test_df.index)].reset_index(drop=True)
#                 trains.append(train_df)
#                 devs.append(dev_df)
#                 tests.append(test_df)
#
#             self._train_df = pd.concat(trains)
#             self._dev_df = pd.concat(devs)
#             self._test_df = pd.concat(tests)
#
#         elif self.task == "mlm":
#             self._train_df = df[~((df["author_id"] == 81) &  (df["work_id"] == 16))].sample(frac=0.7, random_state=self.random_state)
#             self._dev_df = df[~df.index.isin(self._train_df.index) & ~((df["author_id"] == 81) &  (df["work_id"] == 16))]
#             self._test_df = self._dev_df.sample(frac=0.5, random_state=self.random_state)
#             self._dev_df = self._dev_df[~self._dev_df.index.isin(self._test_df.index)]
#
#         else:
#             raise ValueError(f"PandasDataset._split_data() -- Task {self.task} not recognized.")
#
#         # todo: ensure that Dionysius is in the test
#         self._train_df["split"] = "train"
#         self._dev_df["split"] = "dev"
#         self._test_df["split"] = "test"
#
#         df = pd.concat([self._train_df, self._dev_df, self._test_df])
#
#         if self.task == "classification":
#             if not "label" in df.columns:
#                 df["unique_target"] = df.apply(
#                     lambda row: f"{row['target']}_{row['author_id']}" if self.target == "author" and not row["target"] == "Other" else f"{row['target']}",
#                     axis=1
#                 )
#                 df['label'], unique = pd.factorize(df["unique_target"])
#                 df['target'] = df['unique_target']
#                 df.drop(columns=["unique_target"], inplace=True)
#
#                 self.id2label = {idx: label for idx, label in enumerate(unique)}
#             else:
#                 self.id2label = self.get_id2label(df)
#
#         return df


if __name__ == "__main__":
    p = PandasDataset()
    print(p.df)
    print(p.author_metadata_df)
    print(p.df["author_label"])
    print(p.author_metadata_df["epithet"].unique())

    print(p.author_metadata_df.shape)
    print(p.select_authors_by_epithet(["Rhet.", "Orat."]).shape)
    print(p.select_authors_by_epithet(["Rhet.", "Orat."]).columns)
    print(p.select_authors_by_epithet(["Rhet.", "Orat."])["author"].unique())

    # p_dataset = PandasDataset(params=args)
    #print(p_dataset.train_df)
    #print(type(p_dataset.train_df))
    # print(p_dataset.data_df.shape)