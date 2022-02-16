# -- fix path -- Source: https://github.com/KimChengSHEANG/TS_T5
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.data.preprocessing.t5_pre import T5_Preprocessing
from constants import WIKISMALL_TRAIN_ORIGINAL_DATA_PATH, \
    WIKISMALL_VALID_ORIGINAL_DATA_PATH, \
    WIKISMALL_TEST_ORIGINAL_DATA_PATH, \
    WIKISMALL_TRAIN_SIMPLE_DATA_PATH, \
    WIKISMALL_VALID_SIMPLE_DATA_PATH, \
    WIKISMALL_TEST_SIMPLE_DATA_PATH, \
    MODEL_CKPT

if __name__ == "__main__":

    files_path = {"train_original_text_path": WIKISMALL_TRAIN_ORIGINAL_DATA_PATH,
                  "valid_original_text_path": WIKISMALL_VALID_ORIGINAL_DATA_PATH,
                  "test_original_text_path": WIKISMALL_TEST_ORIGINAL_DATA_PATH,
                  "train_simple_text_path": WIKISMALL_TRAIN_SIMPLE_DATA_PATH,
                  "valid_simple_text_path": WIKISMALL_VALID_SIMPLE_DATA_PATH,
                  "test_simple_text_path": WIKISMALL_TEST_SIMPLE_DATA_PATH
                  }

    features = {
        #'WordLengthRatio': {'target_ratio': 0.8},
        #'CharLengthRatio': {'target_ratio': 0.8},
        #'LevenshteinRatio': {'target_ratio': 0.8}
        'DependencyTreeDepthRatio': {'target_ratio': 0.8}
        # 'dependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }

    conf = {
                "model_ckpt" : MODEL_CKPT,
                "type_dataset": "huggingface",
                "files_path": files_path,
                "features": features
    }

    pegasus_pre = T5_Preprocessing(**conf)
    tokens = pegasus_pre.preprocess()
    print(tokens)