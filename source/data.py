from typing import Dict, Optional
from pathlib import Path
import re
import collections
import csv
import datasets
import pandas as pd
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from conf import PREPROCESSED_DIR, FEATURES
from source import features
from source.utils import logging_module

logger = logging_module.get_logger(__name__)


class SimplificationDataModule(LightningDataModule):
    """
  Creating a custom dataset for reading the dataset and
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

    def __init__(
            self,
            model_name: str,
            data_path: Path,
            model_features: Dict,
            split: Optional[str],
            max_seq_length: int = 256,
            train_batch_size: int = 8,
            eval_batch_size: int = 8,
            ):

        super().__init__()
        self.dataset = None
        self.stage = None
        self.split = split
        self.model_name = model_name
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if model_features:
            self.model_features = collections.OrderedDict(sorted(model_features.items()))
    def setup(self, stage: str = "fit"):
        """DataModule pipeline. Load data, add and store features and then tokenize text"""

        self.stage = stage

        if self.stage == "fit":

            logger.info(f"Loading dataset")
            self.dataset = self.load_data()

            path = self._get_path_from_features()
            if not self._exists_preprocessed_dataset(path):
                self._preprocess_training_data()

            logger.info(f"Features calculated previously. Loading preprocessed dataset at: {path}")
            self.dataset["train"] = self._load_preprocessed_dataset(path)
            logger.info("Calculating features")
            self._add_features()

        elif self.stage == "test":

            logger.info(f"Loading dataset")
            self.dataset = self.load_data()

            logger.info("Calculating features")
            self._add_features()
        else:
            raise ValueError("Stage value not supported")

        logger.info("Tokenizing dataset")
        self._tokenize_dataset()

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], batch_size=self.eval_batch_size, num_workers=1)

    def test_dataloader(self):
        if self.split == "validation":
            return DataLoader(self.dataset["valid"], batch_size=self.eval_batch_size, num_workers=1)
        else:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=1)

    def load_data(self):
        """Loading dataset into Hugging Face DatasetDict. simple validation data can included multiple files."""

        if self.stage == "fit":

            valid_original_path = Path(self.data_path / "validation" / "original")
            valid_original_data = pd.concat([pd.read_csv(item, names=["original_text"], sep="\t",
                                                         quoting=csv.QUOTE_NONE)
                                           for item in Path(valid_original_path).glob("*")], axis=1)

            valid_simple_path = Path(self.data_path / "validation" / "simple")
            valid_simple_data = pd.concat([pd.read_csv(item, names=[f"simple_text_valid_{i}"], sep="\t",
                                                       quoting=csv.QUOTE_NONE)
                                           for i,item in enumerate(Path(valid_simple_path).glob("*"))], axis=1)

            valid_data = pd.concat([valid_original_data, valid_simple_data], axis=1)

            path = self._get_path_from_features()
            if path and self._exists_preprocessed_dataset(path):

                dataset_created = datasets.DatasetDict({
                    'valid': datasets.Dataset.from_pandas(valid_data)
                })
            else:
                train_original_path = Path(self.data_path / "train" / "original")

                train_original_data = pd.concat([pd.read_csv(item, names=["original_text"], sep="\t",
                                                             quoting=csv.QUOTE_NONE)
                                                 for item in Path(train_original_path).glob("*")], axis=1)

                train_simple_path = Path(self.data_path / "train" / "simple")
                train_simple_data = pd.concat([pd.read_csv(item, names=[f"simple_text"], sep="\t",
                                                           quoting=csv.QUOTE_NONE)
                                                 for item in Path(train_simple_path).glob("*")], axis=1)

                train_data = pd.concat([train_original_data, train_simple_data], axis=1)

                dataset_created = datasets.DatasetDict({
                    'train': datasets.Dataset.from_pandas(train_data),
                    'valid': datasets.Dataset.from_pandas(valid_data)
                })
        elif self.stage == "test" and self.split == "validation":
            test_original_path = Path(self.data_path / "validation" / "original")
            test_original_data = pd.concat([pd.read_csv(item, names=["original_text"], sep="\t")
                                        for item in Path(test_original_path).glob("*")], axis=1)

            dataset_created = datasets.DatasetDict({
                'valid': datasets.Dataset.from_pandas(test_original_data)
        })


        else:  # self.stage == "test
            test_original_path = Path(self.data_path / "test" / "original")
            test_original_data = pd.concat([pd.read_csv(item, names=["original_text"], sep="\t", quoting=csv.QUOTE_NONE)
                                           for item in Path(test_original_path).glob("*")], axis=1)

            dataset_created = datasets.DatasetDict({
                'test': datasets.Dataset.from_pandas(test_original_data)
            })

        return dataset_created

    def get_features_and_values_string(self):
        string_value = self.model_name + "_"
        if hasattr(self, "model_features"):
            for feature, target in self.model_features.items():
                name = ""
                for word in re.findall('[A-Z][^A-Z]*', feature):
                    if word: name += word[0]
                if not name: name = feature
                string_value += name + "_" + str(target['target_ratio'])
        return string_value

    def get_features_names(self):
        features_names = []
        for feature, _ in self.model_features.items():
            name = ""
            for word in re.findall('[A-Z][^A-Z]*', feature):
                if word: name += word[0]
            if not name: name = feature
            features_names.append(name)
        return features_names

    def _preprocess_training_data(self):

        for feature in FEATURES:
            logger.info(f"Calculating feature: {feature}")
            self.dataset["train"] = self.dataset["train"]. \
                map(getattr(features, feature)("train").get_ratio)
            logger.info(f"Feature: {feature} calculated.")

        logger.info("Storing preprocessed dataset")
        self._store_preprocessed_dataset()


    def _add_features(self):
        """ Calculating features in selected dataset"""

        def _select_specific_train_features(example):

            example['original_text_preprocessed'] = ""
            for feature_name in self.get_features_names():
                example['original_text_preprocessed'] += f'{feature_name}_{example[feature_name]} '
            return example


        def _join_features_and_text(example):

            if example.get('original_text_preprocessed'):
                return {'original_text_preprocessed': 'simplify: ' +
                                                  example['original_text_preprocessed'] +
                                                  example['original_text']}
            else:
                return {'original_text_preprocessed': 'simplify: ' +
                                                  example['original_text']}



        # Selecting split.
        for dataset_split in self.dataset.keys():
            logger.info(f"Calculating split: {dataset_split}")

            # Calculating each feature using map function of Hugging Face dataset calling get_ratio method.
            # dataset_split = "train", "valid" or "test".
            # **kwargs = {"target_ratio"=value}
            if hasattr(self, "model_features"):
                if dataset_split == "train":
                    logger.info(f"Loading training features")
                    self.dataset[dataset_split] = self.dataset[dataset_split].map(_select_specific_train_features)
                    logger.info(f"Training features loaded")
                else:
                    for feature, kwargs in self.model_features.items():
                        logger.info(f"Calculating feature: {feature}")
                        self.dataset[dataset_split] = self.dataset[dataset_split]. \
                                map(getattr(features, feature)(dataset_split, **kwargs).get_ratio)
                        logger.info(f"Feature: {feature} calculated.")

        # Joining features values with the original text
        self.dataset = self.dataset.map(_join_features_and_text)
        if "train" in self.dataset.keys():
            logger.info(f"Training Example:{self.dataset['train'][0]}")
        if "valid" in self.dataset.keys():
            logger.info(f"Validation Example:{self.dataset['valid'][0]}")
        if "test" in self.dataset.keys():
            logger.info(f"Testing Example:{self.dataset['test'][0]}")
        #dataset_split = next(iter(self.dataset.keys()))
        #logger.info(f"Example:{self.dataset[dataset_split][0]}")


    def _tokenize_dataset(self):

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._tokenize_batch,
                fn_kwargs={"split": split},
                batched=True
            )
        if self.stage == "fit":

            columns = ["input_ids", "labels", "attention_mask", "target_mask"]

            self.dataset["train"].set_format(type="torch", columns=columns, output_all_columns=True)
            self.dataset["train"] = self.dataset["train"].map(
                self._replace_pad_token_id)

            columns = ["input_ids", "attention_mask"]

            self.dataset["valid"].set_format(type="torch", columns=columns, output_all_columns=True)

        else:
            columns = ["input_ids", "attention_mask"]

            self.dataset.set_format(type="torch", columns=columns, output_all_columns=True)

    def _tokenize_batch(self, batch, split):

        input_encodings = self.tokenizer(batch["original_text_preprocessed"], max_length=256,
                                         truncation=True, padding="max_length")
        if split == "train":
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(batch["simple_text"], max_length=256,
                                                  truncation=True, padding="max_length")

            return {"original_text": batch["original_text"],
                    "input_ids": input_encodings["input_ids"],
                    "attention_mask": input_encodings["attention_mask"],
                    "simple_text": batch["simple_text"],
                    "labels": target_encodings["input_ids"],
                    "target_mask": target_encodings["attention_mask"]}

        else:

            return {"original_text": batch["original_text"],
                    "input_ids": input_encodings["input_ids"],
                    "attention_mask": input_encodings["attention_mask"]}

    def _replace_pad_token_id(self, batch):

        labels = batch["labels"]
        # Huggingface’s loss functions are defined to exclude the ID -100 during loss calculations.
        # Therefore, we need to convert all padding token IDs in labels to -100.
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    def _exists_preprocessed_dataset(self, path):
        if path:
            return Path(path).exists()

    def _load_preprocessed_dataset(self, path):
        return Dataset.load_from_disk(path)

    def _store_preprocessed_dataset(self):

        path = self._get_path_from_features()
        self.dataset["train"].save_to_disk(path)

    def _get_path_from_features(self):

        path = PREPROCESSED_DIR / self.data_path.name
        return path
