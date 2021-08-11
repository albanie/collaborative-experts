"""This module defines the TextEmbedding interface for converting video descriptions and
queries into embeddings.
"""
import zipfile
import functools
from abc import abstractmethod
from typing import Set, List, Tuple, Union, Callable
from pathlib import Path
from collections import defaultdict

import numpy as np
import spacy
import torch
import gensim
import requests
import transformers
import pkg_resources
import gensim.parsing.preprocessing
from typeguard import typechecked
from symspellpy import SymSpell, Verbosity
from zsvision.zs_utils import BlockTimer

from model.s3dg import S3D


class TextEmbedding:

    def __init__(
            self,
            model: Callable,
            tokenizer: Union[Callable, None],
            dim: int,
            remove_stopwords: bool,
    ):
        self.dim = dim
        self.model = model
        self.tokenizer = tokenizer
        self.remove_stopwords = remove_stopwords
        self.device = None

    @abstractmethod
    def text2vec(self, text: str) -> np.ndarray:
        """Convert a string of text into an embedding.

        Args:
            text: the content to be embedded

        Returns:
            (d x n) array, where d is the dimensionality of the embedding and `n` is the
                number of words that were successfully parsed from the text string.

        NOTE: For some text embedding models (such as word2vec), not all words are
        converted to vectors (e.g. certain kinds of stop words) - these are dropped from
        the output.
        """
        raise NotImplementedError

    @typechecked
    def set_device(self, device: torch.device):
        self.model = self.model.to(device)
        self.device = device


class LookupEmbedding(TextEmbedding):

    @typechecked
    def __init__(
            self,
            model: Callable,
            dim: int,
            remove_stopwords: bool,
            num_samples_for_unknown: int = 50000,
    ):
        tokenizer = Tokenizer(vocab=model.vocab)
        with BlockTimer("generating unknown vector"):
            vecs = np.zeros((min(num_samples_for_unknown, len(model.vocab)), dim))
            for ii, key in enumerate(sorted(model.vocab)):
                if ii >= num_samples_for_unknown:
                    break
                vecs[ii] = model(key)
        self.unknown_vector = np.mean(vecs, 0)

        super().__init__(
            dim=dim,
            model=model,
            tokenizer=tokenizer,
            remove_stopwords=remove_stopwords,
        )

    @typechecked
    def set_device(self, device: torch.device):
        msg = f"{type(self)} only supports CPU-based execution found {device.type}"
        assert device.type == "cpu", msg

    @typechecked
    def text2vec(self, text: str) -> Tuple[np.ndarray, List[str]]:
        if self.remove_stopwords:
            processed_string = gensim.parsing.preprocessing.remove_stopwords(text)
        else:
            processed_string = text
        tokens, failed = self.tokenizer(processed_string)
        embeddings = []
        for token in tokens:
            embeddings.append(self.model(token))
        embeddings = np.array(embeddings)
        msg = (f"Failed to embed any tokens! (text: {text}, processed_string: "
               f"{processed_string}, failed: {failed})")
        if embeddings.size == 0:
            print(f"Warning: {msg}, falling back to unknown vector")
            embeddings = np.array([self.unknown_vector])
        return embeddings, failed


class Tokenizer:
    """For word-level embeddings, we convert words that are absent from the embedding
    lookup table to a canonical tokens (and then re-check the table).  This is to ensure
    that we get reasonable embeddings for as many words as possible.
    """

    @typechecked
    def __init__(self, vocab: Set[str]):
        with BlockTimer("preparing tokenizer dicionaries"):
            # we only use spacy for lemmatising, so we don't need NER or the parser.
            # NOTE: all pronouns are mapped to -PRON-, because it's not clear what their
            # lemma should be (we try to handle these via the spellchecker)
            self.nlp = spacy.load('en', disable=['parser', 'ner'])

            # Symspell is, in theory, a fast spell checker:
            sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt")
            # term_index is the column of the term and count_index is the
            # column of the term frequency
            sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            self.sym_spell = sym_spell
            self.vocab = vocab

            # For a small number of cases, the tokenization fails.
            self.custom = {
                "roundtable": ["round", "table"],
            }

    def __call__(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens, failed = [], []
        for token in doc:
            token, lemma = str(token), token.lemma_
            if token in self.vocab:
                tokens.append(token)
            elif lemma in self.vocab:
                tokens.append(lemma)
            elif lemma in self.custom:
                for subtoken in self.custom[lemma]:
                    if subtoken in self.vocab:
                        tokens.append(subtoken)
                    else:
                        failed.append(subtoken)
            else:
                suggestions = self.sym_spell.lookup(
                    phrase=token,
                    verbosity=Verbosity.CLOSEST,
                    max_edit_distance=2,
                )
                success = False
                for suggestion in suggestions:
                    if suggestion.term in self.vocab:
                        success = True
                        tokens.append(suggestion.term)
                        break
                if not success:
                    failed.append(str(token))
        return tokens, failed


@functools.lru_cache(maxsize=64, typed=False)
def load_w2v_model_from_cache(
        w2v_weights: Path,
) -> gensim.models.keyedvectors.Word2VecKeyedVectors:
    with BlockTimer("Loading w2v from disk"):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fname=w2v_weights,
            binary=True,
        )
    return model


@typechecked
def fetch_model(url: str, weights_path: Path):
    weights_path.parent.mkdir(exist_ok=True, parents=True)
    with BlockTimer(f"Fetching weights {url} -> {weights_path}"):
        resp = requests.get(url, verify=False)
        with open(weights_path, "wb") as f:
            f.write(resp.content)


class W2VEmbedding(LookupEmbedding):
    """This model embeds text using the google-released implementation of the word2vec
    model introduced in:

        Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
        Distributed representations of words and phrases and their compositionality.
        In Advances in neural information processing systems (pp. 3111-3119).

    For words that are present in the w2v vocabulary, a 300-dimensional embedding is
    produced via a lookup table.
    """
    @typechecked
    def __init__(
            self,
            dim: int,
            mirror: str,
            embedding_name: str,
            weights_path: Path,
            remove_stopwords: bool,
            fetch_weights: bool = True,
    ):
        if not weights_path.exists():
            if fetch_weights:
                fetch_model(url=mirror, weights_path=weights_path)
            else:
                raise ValueError(f"w2v weights missing at {weights_path}")

        class W2V_Lookup:
            def __init__(self, w2v):
                self.w2v = w2v
                self.vocab = set(w2v.vocab.keys())

            def __call__(self, key):
                return self.w2v.get_vector(key)

        w2v = load_w2v_model_from_cache(weights_path)
        self.embedding_name = embedding_name
        model = W2V_Lookup(w2v=w2v)

        super().__init__(
            dim=dim,
            model=model,
            remove_stopwords=remove_stopwords,
        )


class GrOVLE(LookupEmbedding):
    """This model wraps various forms of GrOVLE embeddings:

    Args:
        mirror: the URL of a mirror from which the embeddings can be downloaded
        weights_path: where to store the model weights
        embedding_name: one of `grovle`, `mt_grovel`, `hglmm_300d` or `hglmm_6kd`
        fetch_weights: whether to download the weights if they cannot be found on disk.

    NOTES: These embeddings were introduced in the paper:
    Burns, A., Tan, R., Saenko, K., Sclaroff, S., & Plummer, B. A. (2019).
    Language features matter: Effective language representations for vision-language
    tasks. In Proceedings of the IEEE International Conference on Computer Vision
    (pp. 7474-7483).
    """
    @typechecked
    def __init__(
            self,
            dim: int,
            mirror: str,
            embedding_name: str,
            weights_path: Path,
            remove_stopwords: bool,
            fetch_weights: bool = True,
    ):
        if not weights_path.exists():
            if fetch_weights:
                public_url = f"{mirror}/{weights_path.name}"
                fetch_model(url=public_url, weights_path=weights_path)
            else:
                raise ValueError(f"GrOVLE weights missing at {weights_path}")

        with BlockTimer(f"Reading weight contents from {weights_path}"):
            zipped = zipfile.ZipFile(weights_path)
            rows = zipped.read(f"{embedding_name}.txt").decode("utf-8").splitlines()

        # To maintain a consistent interface with w2v, we add a matching `get_vector`
        # method to the dictionary baseclass
        class Lookup(dict):
            def __call__(self, key):
                return self[key]

            @property
            def vocab(self):
                return set(self.keys())

        model = Lookup()

        punctuation = {'"', "?", ")", "("}
        for row in rows:
            # exclude puncutation
            if any([punc in row for punc in punctuation]):
                msg = f"Only expected HGLMM models to have punctuation"
                assert embedding_name in {"hglmm_300d", "hglmm_6kd"}, msg
                continue
            key, weights = row.split(" ", 1)
            model[key] = np.array([float(x) for x in weights.split(" ")])

        super().__init__(
            dim=dim,
            model=model,
            remove_stopwords=remove_stopwords,
        )


class HowTo100M_MIL_NCE(TextEmbedding):
    """This model produces text embeddings trained on HowTo100M using:

    A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic and A. Zisserman, 
    End-to-End Learning of Visual Representations from Uncurated Instructional Videos

    NOTES: This provides a sentence, rather than word-level embedding.
    """

    def __init__(
            self,
            word_dict_path: Path,
            weights_path: Path,
            embedding_name: str,
            dim: int,
            mirror: str,
            fetch_weights: bool = True,
    ):
        for path in [word_dict_path, weights_path]:
            if not path.exists():
                if fetch_weights:
                    public_url = f"{mirror}/{path.name}"
                    fetch_model(url=public_url, weights_path=path)
                else:
                    raise ValueError(f"howto100m weights missing at {path}")

        model = S3D(word_dict_path, dim)
        model.load_state_dict(torch.load(weights_path))
        self.embedding_name = embedding_name
        model.eval()
        super().__init__(
            dim=dim,
            model=model,
            tokenizer=None,
            remove_stopwords=False,
        )

    @typechecked
    def text2vec(self, text: str) -> Tuple[np.ndarray, List[str]]:
        with torch.no_grad():
            embedding = self.model.text_module([text], device=self.device)
        return embedding["text_embedding"].cpu().numpy(), []


class HuggingFaceWrapper(TextEmbedding):
    """This class wraps the embedding of text provided by HuggingFace pretrained models :

    The models can be found here:
    https://huggingface.co/transformers/pretrained_models.html
    """

    def __init__(self, dim: int, embedding_name: str):
        tokenizers = {
            "openai-gpt": transformers.OpenAIGPTTokenizer,
            "bert-base-uncased": transformers.BertTokenizer,
            "ctrl": transformers.CTRLTokenizer,
            "transfo-xl-wt103": transformers.TransfoXLTokenizer,
            "electra": transformers.ElectraTokenizer,
        }
        models = {
            "openai-gpt": transformers.OpenAIGPTModel,
            "bert-base-uncased": transformers.BertModel,
            "ctrl": transformers.CTRLModel,
            "transfo-xl-wt103": transformers.TransfoXLModel,
            "electra": transformers.ElectraModel,
        }
        add_special_tokens = defaultdict(lambda: True)
        add_decoder_input_ids = defaultdict(lambda: False)

        for name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-xl-finetune"]:
            tokenizers[name] = transformers.GPT2Tokenizer
            models[name] = transformers.GPT2Model

        for name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
            tokenizers[name] = transformers.T5Tokenizer
            models[name] = transformers.T5Model
            add_special_tokens[name] = False
            add_decoder_input_ids[name] = True

        for name in ["albert-base-v2", "albert-large-v2", "albert-xlarge-v2"]:
            tokenizers[name] = transformers.AlbertTokenizer
            models[name] = transformers.AlbertModel

        for name in ["roberta-base", "roberta-large"]:
            tokenizers[name] = transformers.RobertaTokenizer
            models[name] = transformers.RobertaModel

        for name in ["xlnet-base-cased", "xlnet-large-cased"]:
            tokenizers[name] = transformers.XLNetTokenizer
            models[name] = transformers.XLNetModel
            add_special_tokens[name] = False

        # handle inconsistent naming scheme for electra
        transformer_keys = {"electra": "google/electra-small-discriminator"}
        transformer_key = transformer_keys.get(embedding_name, embedding_name)
        tokenizer = tokenizers[embedding_name].from_pretrained(transformer_key)
        model = models[embedding_name].from_pretrained(transformer_key)

        self.add_special_tokens = add_special_tokens[embedding_name]
        self.add_decoder_input_ids = add_decoder_input_ids[embedding_name]
        super().__init__(
            model=model,
            dim=dim,
            tokenizer=tokenizer,
            remove_stopwords=False,
        )

    @typechecked
    def text2vec(self, text: str) -> Tuple[np.ndarray, List[str]]:
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
            add_space_before_punct_symbol=True,
        )
        input_idx = torch.LongTensor(tokens).to(self.model.device)
        kwargs = {"input_ids": input_idx.unsqueeze(0)}
        if self.add_decoder_input_ids:
            kwargs["decoder_input_ids"] = input_idx.unsqueeze(0)
        with torch.no_grad():
            hidden_states = self.model(**kwargs)
            embeddings = hidden_states[0].cpu().numpy()
        return embeddings.squeeze(0), []
