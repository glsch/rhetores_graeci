import argparse
import copy
import random
from argparse import ArgumentParser
import os
import time

import numpy as np

from ...datasets.PandasDataset import PandasDataset
from ...path_manager import PathManager
from ...datasets.data_utils import filter_authors
from ...logger_config import logger
from functools import partial
from ...datasets.data_utils import filter_authors, range_sample, add_adversaries, get_candidate_vs_other_split, text2spacy_tokens



import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import warnings

import torch

import torch.functional as F

from transformers import AutoModel, AutoTokenizer

# todo: implement <UNK> in training data
# todo: implement <UNK> in test data
# todo: implement BERT-based baseline
# todo: think how to implement dataset creation
# todo: implement r

def eval_measures(gt, pred, print_flag=True):
    """Compute macro-averaged F1-scores, macro-averaged precision,
    macro-averaged recall, and micro-averaged accuracy according the ad hoc
    rules discussed at the top of this file.
    Parameters
    ----------
    gt : dict
        Ground truth, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    pred : dict
        Predicted attribution, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    Returns
    -------
    f1 : float
        Macro-averaged F1-score
    precision : float
        Macro-averaged precision
    recall : float
        Macro-averaged recall
    accuracy : float
        Micro-averaged F1-score
    """

    actual_authors = list(gt.values())
    encoder = preprocessing.LabelEncoder().fit(['Other'] + actual_authors)
    # encoder = preprocessing.LabelEncoder().fit(actual_authors)

    text_ids, gold_authors, silver_authors = [], [], []

    for text_id in sorted(gt):
        text_ids.append(text_id)
        gold_authors.append(gt[text_id])

        try:
            silver_authors.append(pred[text_id])
        except KeyError:
            # missing attributions get <UNK>:
            silver_authors.append('Other')

    assert len(text_ids) == len(gold_authors)
    assert len(text_ids) == len(silver_authors)

    # replace non-existent silver authors with '<UNK>':
    silver_authors = [a if a in encoder.classes_ else 'Other'
                      for a in silver_authors]

    gold_author_ints = encoder.transform(gold_authors)
    silver_author_ints = encoder.transform(silver_authors)

    # get F1 for individual classes (and suppress warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels=list(set(gold_author_ints))
        # Exclude the <UNK> class
        for x in labels:
            if encoder.inverse_transform([x]) == ['Other']:
                labels.remove(x)
        f1 = metrics.f1_score(y_true=gold_author_ints,
                  y_pred=silver_author_ints,
                  labels=labels,
                  average='macro')
        precision = metrics.precision_score(y_true=gold_author_ints,
                  y_pred=silver_author_ints,
                  labels=labels,
                  average='macro')
        recall = metrics.recall_score(y_true=gold_author_ints,
                  y_pred=silver_author_ints,
                  labels=labels,
                  average='macro')
        accuracy = metrics.accuracy_score(y_true=gold_author_ints,
                  y_pred=silver_author_ints)



        # cm = confusion_matrix(test_labels + ["<UNK>"], predictions, labels=clf.classes_)

        ConfusionMatrixDisplay.from_predictions(gold_author_ints, silver_author_ints,
                                                display_labels=encoder.classes_,
                                                xticks_rotation='vertical')

        plt.show()

    if print_flag:
        print(f"Accuracy: {round(accuracy, 3)}% | Precision: {round(precision, 3)}% | Recall: {round(recall, 3)}% | F1: {round(f1, 3)}%")

    return accuracy, precision, recall, f1

def compute_sim_matrix(feats):
    """
    Takes in a batch of features of size (bs, feat_len).
    """
    sim_matrix = F.cosine_similarity(feats.unsqueeze(2).expand(-1, -1, feats.size(0)),
                                     feats.unsqueeze(2).expand(-1, -1, feats.size(0)).transpose(0, 2),
                                     dim=1)

    return sim_matrix


def compute_target_matrix(labels):
    """
    Takes in a label vector of size (bs)
    """
    label_matrix = labels.unsqueeze(-1).expand((labels.shape[0], labels.shape[0]))
    trans_label_matrix = torch.transpose(label_matrix, 0, 1)
    target_matrix = (label_matrix == trans_label_matrix).type(torch.float)

    return target_matrix

def eval_fn(y_test, y_pred, y_prob=None, average="binary", print_flag=True):

    acc = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
    f1 = round(metrics.f1_score(y_test, y_pred, average=average) * 100, 2)
    recall = round(metrics.recall_score(y_test, y_pred, average=average) * 100, 2)
    precision = round(metrics.precision_score(y_test, y_pred, average=average) * 100, 2)

    if print_flag:
        print(f"Accuracy: {acc}% | Precision: {precision}% | Recall: {recall}% | F1: {f1}%")

    return acc, precision, recall, f1


class MultiInputTransformer(BaseEstimator, TransformerMixin):
    """
    A "decorator"-like transformer to apply a transformer to a specific key of the input DataFrame.

    """
    def __init__(self, transformer, input_key):
        self.transformer = transformer
        self.input_key = input_key

    def fit(self, X, y=None):
        self.transformer.fit(X[self.input_key].tolist(), y)
        return self

    def transform(self, X):
        return self.transformer.transform(X[self.input_key].tolist())

    def get_feature_names_out(self, input_features=None):
        return self.transformer.get_feature_names_out(input_features)

def main():
    parser = ArgumentParser(description="SVM and BERT-based baselines for ")

    parser.add_argument("--dataset_path", type=str, help="Path to the configuration file")
    parser.add_argument("--baseline", type=str, choices=["svm", "bert"], required=True)
    parser.add_argument("--feature", action="append", choices=["3char", "conjugation", "function_words", "3pos", "250mfl"], required=True)
    parser.add_argument("--closed_list", action="store_true", default=False)
    parser.add_argument("--chunk_length", type=int, default=256)
    parser.add_argument("--max_features", type=int, default=1000)
    parser.add_argument("--n_authors", type=int, default=23)
    parser.add_argument("--pt", type=float, default=0.1)
    parser.add_argument("--range", type=float, default=0.5)
    parser.add_argument("--train_r", type=float, default=0.5)
    parser.add_argument("--test_r", default=0.5)
    parser.add_argument("--test_run", action="store_true", default=False)
    parser.add_argument("--transformer", type=str, default='bowphs/LaBerta')
    parser.add_argument("--min_test_samples", type=int, default=25)

    # we assume that in the dataset_path there will be columns text1, text2, target1, target2

    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = os.path.join(PathManager.data_path, "llm_attribution_full.tsv")

    pt = args.pt

    if "250mfl" in args.feature or "3pos" in args.feature:
        tag = True
    else:
        tag = False

    if not os.path.exists(os.path.join(PathManager.data_path, "baseline_df.csv")):

        dataset = PandasDataset.from_file(args.dataset_path, closed=False, tag=tag)
        dataset.data = dataset.data[dataset.data["n_words"] >= args.chunk_length / 2]
        dataset.data = dataset.split_into_chunks(chunk_length=args.chunk_length, sentence_boundaries=True)

        logger.info(f"Dataset loaded from {args.dataset_path}")

        min_samples = None
        if args.test_run:
            min_samples = args.min_test_samples

        dataset.data['row'] = dataset.data.groupby('path').cumcount() + 1
        dataset.data['chunk_id'] = dataset.data.assign(chunk_id=lambda d: d["path"] + " Chunk: " + d["row"].astype(str))[
            "chunk_id"]

        train_df, test_df, calibration_df = dataset.create_problem(df=dataset.data, closed_list=args.closed_list, n_authors=args.n_authors, range=args.range, train_r=args.train_r, test_r=args.test_r, min_samples=min_samples)

        train_df = train_df.assign(split="train")
        test_df = test_df.assign(split="test")
        calibration_df = calibration_df.assign(split="calibration")

        baseline_df = pd.concat([train_df, test_df, calibration_df])
        baseline_df.to_csv(os.path.join(PathManager.data_path, "baseline_df.csv"), index=False)

    else:
        baseline_df = pd.read_csv(os.path.join(PathManager.data_path, "baseline_df.csv"))
        if tag:
            if not all([c in baseline_df.columns for c in ["pos", "lem"]]):
                nlp = spacy.load("la_core_web_lg", exclude=["ner", "parser"])
                baseline_df['text_token'] = baseline_df['text'].progress_apply(
                    partial(text2spacy_tokens, nlp=nlp, serialize=False))
                baseline_df["pos"] = baseline_df["text_token"].progress_apply(lambda x: " ".join([t.pos_ for t in x]))
                baseline_df["lem"] = baseline_df["text_token"].progress_apply(lambda x: " ".join([t.lemma_ for t in x]))
        train_df = baseline_df[baseline_df["split"] == "train"]
        test_df = baseline_df[baseline_df["split"] == "test"]
        calibration_df = baseline_df[baseline_df["split"] == "calibration"]

    print("Train", train_df.shape[0])
    print("Test", test_df.shape[0])

    print("Train counts", train_df.groupby("label").size().sort_values(ascending=False))
    print("Test counts", test_df.groupby("label").size().sort_values(ascending=False))
    print("Calibration counts", calibration_df.groupby("author_name").size().sort_values(ascending=False))

    start_time = time.time()
    logger.info(f"Starting training at {time.ctime()}")

    func_words = ['et', 'in', 'de', 'ad', 'non', 'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                  'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                  'pro', 'autem', 'ibi', 'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                  'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                  'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                  'quidem', 'supra', 'ante', 'adhuc', 'seu', 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                  'postea', 'nunquam']

    latin_conjugations = ['o', 'eo', 'io', 'as', 'es', 'is', 'at', 'et', 'it', 'amus', 'emus', 'imus', 'atis', 'etis',
                          'itis', 'ant', 'ent', 'unt', 'iunt', 'or', 'eor', 'ior', 'aris', 'eris', 'iris', 'atur',
                          'etur',
                          'itur', 'amur', 'emur', 'imur', 'amini', 'emini', 'imini', 'antur', 'entur', 'untur',
                          'iuntur',
                          'abam', 'ebam', 'iebam', 'abas', 'ebas', 'iebas', 'abat', 'ebat', 'iebat', 'abamus', 'ebamus',
                          'iebamus', 'abatis', 'ebatis', 'iebatis', 'abant', 'ebant', 'iebant', 'abar', 'ebar', 'iebar',
                          'abaris', 'ebaris', 'iebaris', 'abatur', 'ebatur', 'iebatur', 'abamur', 'ebamur', 'iebamur',
                          'abamini', 'ebamini', 'iebamini', 'abantur', 'ebantur', 'iebantur', 'abo', 'ebo', 'am', 'iam',
                          'abis', 'ebis', 'ies', 'abit', 'ebit', 'iet', 'abimus', 'ebimus', 'emus', 'iemus', 'abitis',
                          'ebitis', 'ietis', 'abunt', 'ebunt', 'ient', 'abor', 'ebor', 'ar', 'iar', 'aberis', 'eberis',
                          'ieris', 'abitur', 'ebitur', 'ietur', 'abimur', 'ebimur', 'iemur', 'abimini', 'ebimini',
                          'iemini',
                          'abuntur', 'ebuntur', 'ientur', 'i', 'isti', 'it', 'istis', 'erunt', 'em', 'eam', 'eas',
                          'ias', 'eat', 'iat', 'eamus', 'iamus', 'eatis', 'iatis', 'eant', 'iant', 'er', 'ear', 'earis',
                          'iaris', 'eatur', 'iatur', 'eamur', 'iamur', 'eamini', 'iamini', 'eantur', 'iantur', 'rem',
                          'res',
                          'ret', 'remus', 'retis', 'rent', 'rer', 'reris', 'retur', 'remur', 'remini', 'rentur', 'erim',
                          'issem', 'isses', 'isset', 'issemus', 'issetis', 'issent', 'a', 'ate', 'e', 'ete', 'ite',
                          'are',
                          'ere', 'ire', 'ato', 'eto', 'ito', 'atote', 'etote', 'itote', 'anto', 'ento', 'unto', 'iunto',
                          'ator', 'etor', 'itor', 'aminor', 'eminor', 'iminor', 'antor', 'entor', 'untor', 'iuntor',
                          'ari',
                          'eri', 'iri', 'andi', 'ando', 'andum', 'andus', 'ande', 'ans', 'antis', 'anti', 'antem',
                          'antes',
                          'antium', 'antibus', 'antia', 'esse', 'sum', 'est', 'sumus', 'estis', 'sunt', 'eram', 'eras',
                          'erat', 'eramus', 'eratis', 'erant', 'ero', 'eris', 'erit', 'erimus', 'eritis', 'erint',
                          'sim',
                          'sis', 'sit', 'simus', 'sitis', 'sint', 'essem', 'esses', 'esset', 'essemus', 'essetis',
                          'essent',
                          'fui', 'fuisti', 'fuit', 'fuimus', 'fuistis', 'fuerunt', 'este', 'esto', 'estote', 'sunto']

    latin_conjugations = list(set(latin_conjugations))

    if args.baseline == "svm":

        transformer_list = []

        for feature in args.feature:
            if feature == "3char":
                transformer_list.append(("3char", MultiInputTransformer(CountVectorizer(ngram_range=(3, 3), max_features=args.max_features, min_df=10, analyzer="char"), "text")))
            elif feature == "conjugation":
                transformer_list.append(("conjugation", MultiInputTransformer(CountVectorizer(ngram_range=(1, 15), vocabulary=latin_conjugations, analyzer="char"), "text")))
            elif feature == "function_words":
                transformer_list.append(("function_words", MultiInputTransformer(CountVectorizer(ngram_range=(1, 1), vocabulary=func_words, analyzer="word"), "text")))
            elif feature == "250mfl":
                transformer_list.append(("250mfl", MultiInputTransformer(CountVectorizer(ngram_range=(1, 1), max_features=250, analyzer="word"), "lem")))
            elif feature == "3pos":
                transformer_list.append(("3pos", MultiInputTransformer(CountVectorizer(ngram_range=(3, 3), max_features=100, stop_words=["punct"], analyzer="word"), "pos")))

        vectorizer = FeatureUnion(
            transformer_list = transformer_list
        )

        train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()

        train_data = vectorizer.fit_transform(train_df)
        train_data = train_data.astype(float)

        logger.info("Feature union created")

        logger.info(f"Using SVM baseline with {len(vectorizer.get_feature_names_out())}")
        logger.info(f"Used features {vectorizer.get_feature_names_out()} features.")

        for i, v in enumerate(train_texts):
            train_data[i] = train_data[i] / len(train_texts[i])
        # print('\t', 'language: ', language[index])
        # print('\t', len(candidates), 'candidate authors')
        # print('\t', len(train_texts), 'known texts')
        # print('\t', 'vocabulary size:', len(vocabulary))
        # # Building test set
        # test_docs = read_files(path + os.sep + problem, unk_folder)
        # test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_texts = test_df["text"].tolist()
        test_labels = test_df["label"].tolist()
        test_chunks_ids = test_df["chunk_id"].tolist()
        test_data = vectorizer.transform(test_df)
        test_data = test_data.astype(float)

        for i, v in enumerate(test_texts):
            test_data[i] = test_data[i] / len(test_texts[i])

        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        clf = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, class_weight="balanced")), n_jobs=-1)
        logger.info("Fitting SVM")
        clf.fit(scaled_train_data, train_labels)
        logger.info("Done")
        logger.info("Predicting SVM")
        predictions = clf.predict(scaled_test_data)
        logger.info("Done")
        proba = clf.predict_proba(scaled_test_data)

        # Reject option (used in open-set cases)
        count = 0
        predictions = list(predictions)

        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = "Other"
                count = count + 1

        print('\t', count, 'texts left unattributed')

        # for i, v in enumerate(predictions):
        #     print('\t', test_chunks_ids[i], 'Predicted', v, 'Real:', test_labels[i], 'Correct:', v == test_labels[i])

        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f'Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s')

        eval_measures(
            gt={k: v for k, v in zip(test_chunks_ids, test_labels)},
            pred={k: v for k, v in zip(test_chunks_ids, predictions)},
            print_flag=True
        )

        # Saving output data
        # out_data = []
        # #unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        # #pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        # for i, v in enumerate(predictions):
        #     out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        # with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
        #     json.dump(out_data, f, indent=4)
        # print('\t', 'answers saved to file', 'answers-' + problem + '.json')
        # eval_fn(test_labels, predictions, average="weighted", print_flag=True)

    elif args.baseline == "bert":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(args.transformer)
        tokenizer = AutoTokenizer.from_pretrained(args.transformer)

        val_df = test_df.groupby("label").sample(frac=0.5)
        test_df = test_df[~test_df["unique_id"].isin(val_df["unique_id"])]

        texts = train_df["text"].tolist()[:5]

        embeds = tokenizer(text=texts, truncation=True, padding=True, add_special_tokens=False).input_ids

        train_df = train_df.assign(input_ids=train_df["text"].progress_apply(lambda x: tokenizer(x, truncation=True, padding=True, add_special_tokens=False).input_ids))
        token_freq = {}
        for tokens in train_df['input_ids']:
            for token in tokens:
                token_freq[token] = token_freq.setdefault(token, 0) + 1

        feature_freq = {k: v for k, v in sorted(token_freq.items(), key=lambda item: item[1], reverse=True)}
        retained_features = list(feature_freq.keys())[:args.max_features]
        feature_tensor = torch.LongTensor(list([i] for i in range(len(retained_features)))).to(device)

        feature2idx = {f: i for i, f in enumerate(retained_features)}

        logger.debug(f"Feature tensor created: {feature_tensor}")
        logger.debug(f"Feature tensor shape: {feature_tensor.shape}")


        # tokenized_texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=500)
        # embedding = model(tokenized_texts.input_ids.to(model.device),
        #                   tokenized_texts.attention_mask.to(model.device)).last_hidden_state.mean(dim=1)
        # ls_query_text, ls_potential_text = df_sub.loc[i, 'query_text'], df_sub.loc[i, 'potential_text']
        # embed_query_texts = F.normalize(embed_fn(model_name, ls_query_text, baseline_type))
        # embed_potential_texts = F.normalize(embed_fn(model_name, ls_potential_text, baseline_type))
        #
        # preds = embed_query_texts @ embed_potential_texts.T
        # preds = F.softmax(preds, dim=-1)
        # labels = np.arange(0, len(ls_query_text))
        #
        # acc, f1_w, f1_micro, f1_macro = eval_fn(labels, preds.argmax(-1).numpy())
        # ls_acc.append(acc)
        # ls_f1_w.append(f1_w)
        # ls_f1_micro.append(f1_micro)
        # ls_f1_macro.append(f1_macro)


    # X = vectorizer.fit_transform(dataset.data["text"])
    # print("\nVectorized Text (first 5 features):\n", X.toarray()[:, :5])


if __name__ == "__main__":
    main()