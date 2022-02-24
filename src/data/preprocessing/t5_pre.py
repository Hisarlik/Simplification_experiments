from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from datasets import Dataset
from src.utils import logging_module
from src.data.datasets import HuggingFaceDataset
from src.data.preprocessing import PreprocessingBase
from src.features import features

logger = logging_module.get_logger(__name__)


class T5_Preprocessing(PreprocessingBase):

    def __init__(self, **kwargs):

        self.type_dataset = HuggingFaceDataset(kwargs.get("files_path"))
        self.features = kwargs.get("features")
        self.kwargs = kwargs

        model_ckpt = kwargs.get('model_ckpt')
        if model_ckpt:
            self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def preprocess(self):
        """
        Main method to preprocess the dataset adding the selected features.
        Returns:
            'dataset': Hugging Face dataset object with added features.
        """
        logger.info("Loading dataset...")
        dataset = self.load_data()
        logger.info("Dataset loaded")

        for feature, kwargs in self.features.items():
            logger.info(f"Calculating feature: {feature}")
            dataset = dataset.map(getattr(features, feature)().get_ratio,
                                  fn_kwargs=kwargs)
            logger.info(f"Feature: {feature} calculated.")

        dataset = dataset.map(lambda example: {'original_text_preprocessed': example['original_text_preprocessed'] +
                                                                                 example['original_text']})
        print(dataset["train"][0])
        self.save_data(dataset)
        #dataset = dataset["train"].to_pandas()
        #return dataset

    def load_data(self, **kwargs):
        return self.type_dataset.load()

    def save_data(self, dataset: Dataset):

        self.type_dataset.save(dataset,
                               Path(self.kwargs.get("preprocessed_data_path")))

    def load_from_disk(self, path):
        return self.type_dataset.load_from_disk(path)

    def tokenize(self, dataset: Dataset):
        dataset_samsum_pt = dataset.map(self._tokenize_batch,
                                        batched=True)
        columns = ["input_ids", "labels", "attention_mask"]
        dataset_samsum_pt.set_format(type="torch", columns=columns)
        return dataset_samsum_pt

    def _tokenize_batch(self, batch):
        input_encodings = self.tokenizer(batch["original_text"], max_length=32,
                                         truncation=True, padding="max_length")

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(batch["simple_text"], max_length=32,
                                              truncation=True, padding="max_length")

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}
