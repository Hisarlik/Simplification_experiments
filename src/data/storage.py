import pathlib
from typing import List, Tuple
import pickle
import pandas as pd
from src.utils import logging_module

logger = logging_module.get_logger(__name__)


def load_data_from_files(
        original_text_path: pathlib.Path, simple_text_path: pathlib.Path
) -> Tuple[List, List]:

    with open(original_text_path, "r", encoding="utf8") as f:
        original_texts = f.readlines()

    with open(simple_text_path, "r", encoding="utf8") as f:
        simple_texts = f.readlines()

    assert len(original_texts) == len(simple_texts)

    return original_texts, simple_texts


def store_df_2file(df: pd.DataFrame, output_path: pathlib.Path):
    pathlib.Path(output_path).parents[0].mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as fp:
        pickle.dump(df, fp)
        logger.info(f"Preprocessed Dataset stored in a Pandas dataFrame: {output_path}")
