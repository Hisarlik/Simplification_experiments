import torch

from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
DUMPS_DIR = RESOURCES_DIR / "DUMPS"
WORD_EMBEDDINGS_NAME = "glove.42B.300d"

##### WIKISMALL #####
WIKISMALL_DATASET = RESOURCES_DIR / "wikismall"

WIKISMALL_TRAIN_ORIGINAL_DATA_PATH = WIKISMALL_DATASET / "PWKP_108016.tag.80.aner.ori.train.src"
WIKISMALL_VALID_ORIGINAL_DATA_PATH = WIKISMALL_DATASET / "PWKP_108016.tag.80.aner.ori.valid.src"
WIKISMALL_TEST_ORIGINAL_DATA_PATH = WIKISMALL_DATASET / "PWKP_108016.tag.80.aner.ori.test.src"
WIKISMALL_TRAIN_SIMPLE_DATA_PATH = WIKISMALL_DATASET / "PWKP_108016.tag.80.aner.ori.train.dst"
WIKISMALL_VALID_SIMPLE_DATA_PATH = WIKISMALL_DATASET / "PWKP_108016.tag.80.aner.ori.valid.dst"
WIKISMALL_TEST_SIMPLE_DATA_PATH = WIKISMALL_DATASET / "PWKP_108016.tag.80.aner.ori.test.dst"



##### PEGASUS #####
MODEL_CKPT = "google/pegasus-cnn_dailymail"


##### T5 #####


####### EditNTS #########################
VOCAB_POS_TAGGING_PATH = RESOURCES_DIR / "resources/editnts/postag_set.p"
VOCAB_WORDS_PATH = RESOURCES_DIR / "resources/editnts/vocab.txt"
OUTPUT_PREPROCESSED_DATA = RESOURCES_DIR / "models/editnts/wikismall/preprocessed/preprocessed_df"


PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

PAD_ID = 0 #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK_ID = 1 #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP_ID = 2 # This has a vocab id, which is used for copying from the source [2]
DEL_ID = 3 # This has a vocab id, which is used for deleting the corresponding word [3]
START_ID = 4 # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP_ID = 5 # This has a vocab id, which is used to stop decoding [5]



##### DEVICE #####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
