from abc import ABC, abstractmethod
import re
from sacremoses import MosesTokenizer
import Levenshtein
import spacy
import nltk
import pickle
import urllib
import os
import tarfile
import zipfile
from tqdm import tqdm
from pathlib import Path
import numpy as np
from string import punctuation
import transformers
import torch

nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk
from conf import DUMPS_DIR, WORD_EMBEDDINGS_NAME

stopwords = set(stopwords.words("english"))


def ControlDivisionByZero(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


class FeatureAbstract(ABC):

    @abstractmethod
    def get_ratio(self, kwargs):
        pass

    @abstractmethod
    def calculate_ratio(self, simple_text, original_text):
        pass


class Feature(FeatureAbstract):

    def __init__(self, split, target_ratio):
        self.split = split
        self.target_ratio = target_ratio

    def get_ratio(self, kwargs):
        if not 'original_text_preprocessed' in kwargs:
            kwargs['original_text_preprocessed'] = ""

        if self.split == "train":
            simple_text = kwargs.get('simple_text')
            original_text = kwargs.get('original_text')
            result_ratio = self.calculate_ratio(simple_text, original_text)
            kwargs[self.name] = result_ratio

        elif self.split == "valid" or self.split == "test":
            result_ratio = self.target_ratio
            kwargs['original_text_preprocessed'] += f'{self.name}_{result_ratio} '
        else:
            raise ValueError("stage value not supported")

        return kwargs

    @property
    def name(self):
        class_name = self.__class__.__name__
        name = ""
        for word in re.findall('[A-Z][^A-Z]*', class_name):
            if word: name += word[0]
        if not name: name = class_name
        return name


class WordLengthRatio(Feature):

    def __init__(self, stage="train", target_ratio=0.7):
        super().__init__(stage, target_ratio)
        if stage == "train":
            self.tokenizer = MosesTokenizer(lang='en')

    def calculate_ratio(self, simple_text, original_text):
        return round(ControlDivisionByZero(
            len(self.tokenizer.tokenize(simple_text)),
            len(self.tokenizer.tokenize(original_text))), 2)


class CharLengthRatio(Feature):

    def __init__(self, stage="train", target_ratio=0.7):
        super().__init__(stage, target_ratio)

    def calculate_ratio(self, simple_text, original_text):
        return round(ControlDivisionByZero(
            len(simple_text),
            len(original_text)), 2)


class LevenshteinRatio(Feature):

    def __init__(self, stage="train", target_ratio=0.7):
        super().__init__(stage, target_ratio)

    def calculate_ratio(self, simple_text, original_text):
        return round(Levenshtein.ratio(original_text,
                                       simple_text), 2)


class DependencyTreeDepthRatio(Feature):

    def __init__(self, stage="train", target_ratio=0.7):
        super().__init__(stage, target_ratio)
        if stage == "train":
            self.nlp = self.get_spacy_model()

    def get_spacy_model(self):

        model = 'en_core_web_sm'
        if not spacy.util.is_package(model):
            spacy.cli.download(model)
            spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
        return spacy.load(model)

    def calculate_ratio(self, simple_text, original_text):

        result_ratio = round(ControlDivisionByZero(
            self.get_dependency_tree_depth(simple_text),
            self.get_dependency_tree_depth(original_text)), 2)

        return result_ratio

    def get_dependency_tree_depth(self, sentence):

        def get_subtree_depth(node):
            if len(list(node.children)) == 0:
                return 0
            return 1 + max([get_subtree_depth(child) for child in node.children])

        tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in self.nlp(sentence).sents]
        if len(tree_depths) == 0:
            return 0
        return max(tree_depths)


class WordRankRatio(Feature):

    def __init__(self, stage="train", target_ratio=0.7):
        super().__init__(stage, target_ratio)
        if stage == "train":
            self.tokenizer = MosesTokenizer(lang='en')
            self.word2rank = self._get_word2rank()
            self.length_rank = len(self.word2rank)

    def calculate_ratio(self, simple_text, original_text):

        result_ratio = round(min(ControlDivisionByZero(self.get_lexical_complexity_score(simple_text),
                                                       self.get_lexical_complexity_score(original_text)),
                                 2), 2)

        return result_ratio

    def get_lexical_complexity_score(self, sentence):

        words = self.tokenizer.tokenize(self._remove_stopwords(self._remove_punctuation(sentence)))
        words = [word for word in words if word in self.word2rank]
        if len(words) == 0:
            return np.log(1 + self.length_rank)
        return np.quantile([self._get_rank(word) for word in words], 0.75)

    def _remove_punctuation(self, text):
        return ' '.join([word for word in self.tokenizer.tokenize(text) if not self._is_punctuation(word)])

    def _remove_stopwords(self, text):
        return ' '.join([w for w in self.tokenizer.tokenize(text) if w.lower() not in stopwords])

    def _is_punctuation(self, word):
        return ''.join([char for char in word if char not in punctuation]) == ''

    def _get_rank(self, word):
        rank = self.word2rank.get(word, self.length_rank)
        return np.log(1 + rank)

    def _get_word2rank(self, vocab_size=np.inf):
        model_filepath = DUMPS_DIR / f"{WORD_EMBEDDINGS_NAME}.pk"
        if model_filepath.exists():
            with open(model_filepath, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            print("Downloading glove.42B.300d ...")
            self._download_glove(model_name='glove.42B.300d', dest_dir=str(DUMPS_DIR))
            print("Preprocessing word2rank...")
            DUMPS_DIR.mkdir(parents=True, exist_ok=True)
            WORD_EMBEDDINGS_PATH = DUMPS_DIR / f'{WORD_EMBEDDINGS_NAME}.txt'
            lines_generator = self._yield_lines(WORD_EMBEDDINGS_PATH)
            word2rank = {}
            # next(lines_generator)
            for i, line in enumerate(lines_generator):
                if i >= vocab_size: break
                word = line.split(' ')[0]
                word2rank[word] = i

            pickle.dump(word2rank, open(model_filepath, 'wb'))
            txt_file = DUMPS_DIR / f'{WORD_EMBEDDINGS_NAME}.txt'
            zip_file = DUMPS_DIR / f'{WORD_EMBEDDINGS_NAME}.zip'
            if txt_file.exists(): txt_file.unlink()
            if zip_file.exists(): zip_file.unlink()
            return word2rank

    def _download_glove(self, model_name, dest_dir):
        url = ''
        if model_name == 'glove.6B':
            url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        elif model_name == 'glove.42B.300d':
            url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
        elif model_name == 'glove.840B.300d':
            url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
        elif model_name == 'glove.twitter.27B':
            url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        else:
            possible_values = ['glove.6B', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B']
            raise ValueError('Unknown model_name. Possible values are {}'.format(possible_values))
        file_path = self._download_url(url, dest_dir)
        out_filepath = Path(file_path)
        out_filepath = out_filepath.parent / f'{out_filepath.stem}.txt'
        # print(out_filepath, out_filepath.exists())
        if not out_filepath.exists():
            print("Extracting: ", Path(file_path).name)
            self._unzip(file_path, dest_dir)

    def _yield_lines(self, filepath):
        filepath = Path(filepath)
        with filepath.open('r', encoding="latin-1") as f:
            for line in f:
                yield line.rstrip()

    def _download_url(self, url, output_path):
        name = url.split('/')[-1]
        file_path = f'{output_path}/{name}'
        if not Path(file_path).exists():
            with tqdm(unit='B', unit_scale=True, leave=True, miniters=1,
                      desc=name) as t:  # all optional kwargs
                urllib.request.urlretrieve(url, filename=file_path, reporthook=self._download_report_hook(t), data=None)
        return file_path

    def _unzip(self, file_path, dest_dir=None):
        if dest_dir is None:
            dest_dir = os.path.dirname(file_path)
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(dest_dir)
            tar.close()
        elif file_path.endswith("tar"):
            tar = tarfile.open(file_path, "r:")
            tar.extractall(dest_dir)
            tar.close()

    def _download_report_hook(self, t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner


class LMFillMaskRatio(Feature):

    def __init__(self, stage="train", target_ratio=0.7):
        super().__init__(stage, target_ratio)
        if stage == "train":
            self.model = transformers.AutoModelWithLMHead.from_pretrained('lordtt13/COVID-SciBERT')
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('lordtt13/COVID-SciBERT')

    def calculate_ratio(self, simple_text, original_text):
        complex_feature = self.mask_sentence_prediction(original_text)
        simple_feature = self.mask_sentence_prediction(simple_text)
        if complex_feature != 0:
            value = simple_feature / complex_feature
        else:
            value = 1
        return round(value, 2)

    def mask_sentence_prediction(self, text, topk=50):

        sentence_tokens = [token[0] for token in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        pos_tagging_sentence = pos_tag(sentence_tokens)
        predictions = []
        for i, (word, pos) in enumerate(pos_tagging_sentence):

            if pos in ['NNS', 'NN', 'VBP', 'VBG', 'VBD']:
                sentence_masked = sentence_tokens.copy()
                sentence_masked[i] = "[MASK]"
                sentence_final = " ".join(sentence_masked)
                inputs = self.tokenizer(sentence_final, return_tensors="pt")
                token_logits = self.model(**inputs).logits
                mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
                mask_token_logits = token_logits[0, mask_token_index, :]
                top_n_tokens = torch.topk(mask_token_logits, topk, dim=1).indices[0].tolist()
                predicted_tokens = [self.tokenizer.decode([token]) for token in top_n_tokens]
                if word in predicted_tokens:
                    predictions.append(predicted_tokens.index(word))

        predictions = [np.log(pred + 1) for pred in predictions]
        if len(predictions) == 0:
            return np.log(1 + topk*50)

        value = np.quantile(predictions, 0.75)
        return value
