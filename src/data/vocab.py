import pickle
import numpy as np
from src.constants import PAD, UNK, KEEP, DEL, START, STOP, VOCAB_POS_TAGGING_PATH
from src.utils import logging_module

logger = logging_module.get_logger(__name__)

class Vocab():
    def __init__(self):
        self.word_list = [PAD, UNK, KEEP, DEL, START, STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def add_vocab_from_file(self, vocab_file="../vocab_data/vocab.txt",vocab_size=30000):
        with open(vocab_file, "r") as f:
            for i,line in enumerate(f):
                if i >=vocab_size:
                    break
                self.word_list.append(line.split()[0])  # only want the word, not the count
        logger.info(f"read %d words from vocab file: {len(self.word_list)}")

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile="path_for_glove_embedding", embed_size=100):
        logger.info("Loading Glove embeddings")
        with open(gloveFile, 'r') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))
            count = 0
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    embedding_matrix[self.w2i[word]] = embedding
                    count += 1

        self.embedding = embedding_matrix
        logger.info(f"{count} words out of {len(self.word_list)} has embeddings in the glove file")

class POSvocab():
    def __init__(self, vocab_pos_path : str = VOCAB_POS_TAGGING_PATH):
        self.word_list = [PAD,UNK,START,STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None
        with open(vocab_pos_path,'rb') as f:
            # postag_set is from NLTK
            tagdict = pickle.load(f)

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

        for w in tagdict:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1