from src.data.preprocessing import pegasus_pre
from src.models import pegasus_sum
import torch
from constants import TRAIN_ORIGINAL_DATA_PATH, \
                      VALID_ORIGINAL_DATA_PATH, \
                      TEST_ORIGINAL_DATA_PATH,  \
                      TRAIN_SIMPLE_DATA_PATH,   \
                      VALID_SIMPLE_DATA_PATH,   \
                      TEST_SIMPLE_DATA_PATH,    \
                      MODEL_CKPT



if __name__ == "__main__":

    files_path = {"train_original_text_path": TRAIN_ORIGINAL_DATA_PATH,
                  "valid_original_text_path": VALID_ORIGINAL_DATA_PATH,
                  "test_original_text_path": TEST_ORIGINAL_DATA_PATH,
                  "train_simple_text_path": TRAIN_SIMPLE_DATA_PATH,
                  "valid_simple_text_path": VALID_SIMPLE_DATA_PATH,
                  "test_simple_text_path": TEST_SIMPLE_DATA_PATH,
                  "type_dataset": "huggingface",
                  "model_ckpt": MODEL_CKPT
                  }

    pegasus = pegasus_pre.PreprocessingPegasus(**files_path)
    tokens = pegasus.pipeline()
    print(tokens)
    model = pegasus_sum.PegasusSumModel(pegasus.tokenizer)
    model.train(**tokens)
