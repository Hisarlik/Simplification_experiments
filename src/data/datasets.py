from pathlib import Path

from datasets import Dataset, DatasetDict
import pandas as pd

class HuggingFaceDataset():

    def __init__(self, kwargs):

        self.train_original_text_path = kwargs.get('train_original_text_path')
        self.valid_original_text_path = kwargs.get('valid_original_text_path')
        self.test_original_text_path = kwargs.get('test_original_text_path')
        self.train_simple_text_path = kwargs.get('train_simple_text_path')
        self.valid_simple_text_path = kwargs.get('valid_simple_text_path')
        self.test_simple_text_path = kwargs.get('test_simple_text_path')

    def load(self):
        train_original_data = pd.read_csv(self.train_original_text_path, sep="\t", header=None, names=["original_text"])
        valid_original_data = pd.read_csv(self.valid_original_text_path, sep="\t", header=None, names=["simple_text"])
        test_original_data = pd.read_csv(self.test_original_text_path, sep="\t", header=None, names=["original_text"])
        train_simple_data = pd.read_csv(self.train_simple_text_path, sep="\t", header=None, names=["simple_text"])
        valid_simple_data = pd.read_csv(self.valid_simple_text_path, sep="\t", header=None, names=["original_text"])
        test_simple_data = pd.read_csv(self.test_simple_text_path, sep="\t", header=None, names=["simple_text"])

        train_data = pd.concat([train_original_data, train_simple_data], axis=1)
        valid_data = pd.concat([valid_original_data, valid_simple_data], axis=1)
        test_data = pd.concat([test_original_data, test_simple_data], axis=1)

        train_test_valid_dataset = DatasetDict({
            'train': Dataset.from_pandas(train_data),
            'valid': Dataset.from_pandas(valid_data),
            'test': Dataset.from_pandas(test_data)
        })

        return train_test_valid_dataset


    def save(self, dataset: Dataset, path: Path):
        path.mkdir(exist_ok=True)
        dataset.save_to_disk(str(path))


    def load_from_disk(self, path):
        return DatasetDict.load_from_disk(path)





