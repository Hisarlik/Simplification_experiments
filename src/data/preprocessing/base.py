from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd


class PreprocessingBase(ABC):

    @abstractmethod
    def load_data(self):
        pass

    # @abstractmethod
    # def clean_data(self):
    #     pass

    @abstractmethod
    def tokenize(self):
        pass

    # @abstractmethod
    # def build_vocabulary(self):
    #     pass

    # @abstractmethod
    # def mapping_and_padding(self):
    #     pass