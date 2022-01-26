from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from nltk import pos_tag
from pandas.core.frame import DataFrame

from src.utils import logging_module
from src.editnts.data import storage, vocabulary
from src.editnts.constants import UNK,VOCAB_POS_TAGGING_PATH, OUTPUT_PREPROCESSED_DATA,VOCAB_WORDS_PATH

logger = logging_module.get_logger(__name__)

def preprocess_raw_data(
    original_text_path: Path, simple_text_path: Path
) -> pd.DataFrame:

    original_txts, simple_txts = storage.load_data_from_files(
        original_text_path, simple_text_path
    )
    logger.debug("files loaded")

    df = pd.DataFrame(
        {
            "comp_tokens": _tokenize(original_txts),
            "simp_tokens": _tokenize(simple_txts),
        }
    )  
    logger.debug("text tokenized")   
    
    df = _add_pos(df)
    logger.debug("pos tagging added")
    
    df = _add_edits(df)
    logger.debug("edit operations field added")

    _editnet_data_to_editnetID(df)
    logger.info("Add vocab columns to df and store it")

    return df

def _editnet_data_to_editnetID(df: pd.DataFrame):
    """
    this function reads from df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    and add vocab ids for comp_tokens, simp_tokens, and edit_labels
    :param df: df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    :param output_path: the path to store the df
    :return: a dataframe with df.columns=['comp_tokens', 'simp_tokens', 'edit_labels',
                                            'comp_ids','simp_id','edit_ids',
                                            'comp_pos_tags','comp_pos_ids'])
    """
    out_list = []
    vocab = vocabulary.Vocab()
    vocab.add_vocab_from_file(VOCAB_WORDS_PATH, 30000)

    def prepare_example(example, vocab):
        """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens', 'edit_labels']
        :param vocab: vocab object for translation
        :return: inp: original input sentence,
        """
        comp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['comp_tokens']])
        simp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['simp_tokens']])
        edit_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['edit_labels']])
        return comp_id, simp_id, edit_id  # add a dimension for batch, batch_size =1

    for i,example in df.iterrows():

        comp_id, simp_id, edit_id = prepare_example(example,vocab)
        ex=[example['comp_tokens'], comp_id,
         example['simp_tokens'], simp_id,
         example['edit_labels'], edit_id,
         example['comp_pos_tags'],example['comp_pos_ids']
         ]
        out_list.append(ex)
    outdf = pd.DataFrame(out_list, columns=['comp_tokens','comp_ids', 'simp_tokens','simp_ids',
                                            'edit_labels','new_edit_ids','comp_pos_tags','comp_pos_ids'])

    storage.store_df_2file(outdf, Path(OUTPUT_PREPROCESSED_DATA))


def _tokenize(texts: List[str]) -> List[List[str]]:
    return [line.lower().split() for line in texts]


def _add_pos(df: DataFrame) -> DataFrame:
    src_sentences = df["comp_tokens"].tolist()
    pos_sentences = [pos_tag(sent) for sent in src_sentences]
    df["comp_pos_tags"] = pos_sentences

    pos_vocab = vocabulary.POSvocab(VOCAB_POS_TAGGING_PATH)
    pos_ids_list = []
    for sent in pos_sentences:
        pos_ids = [
            pos_vocab.w2i[w[1]] if w[1] in pos_vocab.w2i.keys() else pos_vocab.w2i[UNK]
            for w in sent
        ]
        pos_ids_list.append(pos_ids)
    df["comp_pos_ids"] = pos_ids_list
    return df

def _add_edits(df: DataFrame) -> DataFrame:
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        pair_sentences = list(zip(comp_sentences,simp_sentences))

        edits_list = [_sent2edit(l[0],l[1]) for l in pair_sentences] # transform to edits based on comp_tokens and simp_tokens
        df['edit_labels'] = edits_list
        return df

def _sent2edit(sent1: List[str], sent2: List[str]) -> List[List[str]]:


    dp = _edit_distance(sent1, sent2)
    edits = []
    pos = []
    m, n = len(sent1), len(sent2)
    while m != 0 or n != 0:
        curr = dp[m][n]
        if m==0: #have to insert all here
            while n>0:
                left = dp[1][n-1]
                edits.append(sent2[n-1])
                pos.append(left)
                n-=1
        elif n==0:
            while m>0:
                top = dp[m-1][n]
                edits.append('DEL')
                pos.append(top)
                m -=1
        else: # we didn't reach any special cases yet
            diag = dp[m-1][n-1]
            left = dp[m][n-1]
            top = dp[m-1][n]
            if sent2[n-1] == sent1[m-1]: # keep
                edits.append('KEEP')
                pos.append(diag)
                m -= 1
                n -= 1
            elif curr == top+1: # INSERT preferred before DEL
                edits.append('DEL')
                pos.append(top)  # (sent2[n-1])
                m -= 1
            else: #insert
                edits.append(sent2[n - 1])
                pos.append(left)  # (sent2[n-1])
                n -= 1
    edits = edits[::-1]
    # replace the keeps at the end to stop, this helps a bit with imbalanced classes (KEEP,INS,DEL,STOP)
    for i in range(len(edits))[::-1]: #reversely checking
        if edits[i] == 'KEEP':
            if edits[i-1] =='KEEP':
                edits.pop(i)
            else:
                edits[i] = 'STOP'
                break
    # if edits == []: # do we learn edits if input and output are the same?
    #     edits.append('STOP') #in the case that input and output sentences are the same
    return edits


def _edit_distance(sent1:List[str], sent2:List[str], max_id: int=4999) -> List[List[int]]:
    # edit from sent1 to sent2
    # Create a table to store results of subproblems
    m = len(sent1)
    n = len(sent2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):
            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif sent1[i-1] == sent2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                edit_candidates = np.array([
                    dp[i][j-1], # Insert
                    dp[i-1][j] # Remove
                    ])
                dp[i][j] = 1 + min(edit_candidates)
    return dp

