from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from src.data.datasets import HuggingFaceDataset
from src.data.preprocessing import PreprocessingBase

class PreprocessingPegasus(PreprocessingBase):

    def __init__(self, **kwargs):

        self.type_dataset = HuggingFaceDataset(kwargs)
        self.dataset = None
        model_ckpt = kwargs.get('model_ckpt')
        if model_ckpt:
            self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def pipeline(self):
        self.load_data()
        dataset_tokenized = self.tokenize()
        return dataset_tokenized

    def load_data(self, **kwargs):
        self.dataset = self.type_dataset.load()


    def tokenize(self):

        dataset_samsum_pt = self.dataset.map(self._tokenize_batch,
                                               batched=True)
        columns = ["input_ids", "labels", "attention_mask"]
        dataset_samsum_pt.set_format(type="torch", columns=columns)
        return dataset_samsum_pt


    def _tokenize_batch(self, batch):

        input_encodings = self.tokenizer(batch["original_text"], max_length=1024,
                                    truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(batch["simple_text"], max_length=128,
                                         truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}


