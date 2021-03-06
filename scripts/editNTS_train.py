import sys
sys.dont_write_bytecode = True
import torch
import collections
import argparse
from pathlib import Path
from constants import DEVICE, WIKISMALL_TRAIN_ORIGINAL_DATA_PATH, WIKISMALL_TRAIN_SIMPLE_DATA_PATH
from src.utils import logging_module
from src.models import editnts
from src.data import vocabulary
from src.data.preprocessing import editnts

logger = logging_module.get_logger(__name__)





def main():
    torch.manual_seed(233)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_preprocessed_path', type=str, dest='data_preprocessed_path',
                        default='models/editnts/wikismall/preprocessed/preprocessed_df',
                        help='Path to train vocab_data')
    parser.add_argument('--store_dir', action='store', dest='store_dir',
                        default='models/editnts/wikismall/tmp_store/editNTS',
                        help='Path to exp storage directory.')
    parser.add_argument('--vocab_path', type=str, dest='vocab_path',
                        default='resources/editnts/',
                        help='Path contains vocab, embedding, postag_set')
    parser.add_argument('--vocab_size', dest='vocab_size', default=30000, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--max_seq_len', dest='max_seq_len', default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    vocab = vocabulary.Vocab()
    vocab.add_vocab_from_file(args.vocab_path+'vocab.txt', args.vocab_size)
    vocab.add_embedding(gloveFile=args.vocab_path+'glove.6B.100d.txt')
    pos_vocab = vocabulary.POSvocab(args.vocab_path + 'postag_set.p')

    hyperparams=collections.namedtuple(
        'hps',
        ['vocab_size', 'embedding_dim',
         'word_hidden_units', 'sent_hidden_units',
         'pretrained_embedding', 'word2id', 'id2word',
         'pos_vocab_size', 'pos_embedding_dim','n_layers']
    )
    hps = hyperparams(
        vocab_size=vocab.count,
        embedding_dim=100,
        word_hidden_units=args.hidden,
        sent_hidden_units=args.hidden,
        pretrained_embedding=vocab.embedding,
        word2id=vocab.w2i,
        id2word=vocab.i2w,
        pos_vocab_size=pos_vocab.count,
        pos_embedding_dim=30,
        n_layers=1
    )

    logger.info('init editNTS model')
    edit_net = editnts.EditNTSModel(hps).to(DEVICE)
    edit_net()

if __name__ == "__main__":

    files_path = {"original_text_path": WIKISMALL_TRAIN_ORIGINAL_DATA_PATH,
                  "simple_text_path": WIKISMALL_TRAIN_SIMPLE_DATA_PATH}

    prepro = editnts.PreprocessingEditNTS(**files_path)
    prepro.preprocess()
