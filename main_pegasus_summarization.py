from src.data.preprocessing import pegasus_pre
from src.models import pegasus
import torch
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
                  "test_simple_text_path": WIKISMALL_TEST_SIMPLE_DATA_PATH,
                  "type_dataset": "huggingface",
                  "model_ckpt": MODEL_CKPT
                  }

    pegasus_pre = pegasus_pre.PreprocessingPegasus(**files_path)
    tokens = pegasus_pre.preprocess()
    print(tokens)
    model = pegasus.PegasusSumModel(pegasus_pre.tokenizer)
    #model.train(**tokens)
