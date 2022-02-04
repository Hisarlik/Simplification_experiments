import torch

TRAIN_ORIGINAL_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.train.src"
VALID_ORIGINAL_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.valid.src"
TEST_ORIGINAL_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.test.src"
TRAIN_SIMPLE_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.train.dst"
VALID_SIMPLE_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.valid.dst"
TEST_SIMPLE_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.test.dst"



##### PEGASUS ########################
MODEL_CKPT = "google/pegasus-cnn_dailymail"





####### EditNTS #########################
VOCAB_POS_TAGGING_PATH = "resources/editnts/postag_set.p"
VOCAB_WORDS_PATH = "resources/editnts/vocab.txt"
OUTPUT_PREPROCESSED_DATA = "models/editnts/wikismall/preprocessed/preprocessed_df"


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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"