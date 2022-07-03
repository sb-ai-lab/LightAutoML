"""Tokenizer classes for text preprocesessing and tokenization."""

import re

from functools import partial
from multiprocessing import Pool
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union


try:
    import nltk

    from nltk.stem import SnowballStemmer
except:
    import warnings

    warnings.warn("'nltk' - package isn't installed")


from ..dataset.base import RolesDict
from ..dataset.roles import ColumnRole


Roles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]


def tokenizer_func(arr, tokenizer):
    """Additional tokenizer function."""
    return [tokenizer._tokenize(x) for x in arr]


class BaseTokenizer:
    """Base class for tokenizer method."""

    _fname_prefix = None
    _fit_checks = ()
    _transform_checks = ()

    def __init__(self, n_jobs: int = 4, to_string: bool = True, **kwargs: Any):
        """Tokenization with simple text cleaning and preprocessing.

        Args:
            n_jobs: Number of threads for multiprocessing.
            to_string: Return string or list of tokens.

        """
        self.n_jobs = n_jobs
        self.to_string = to_string

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        return snt

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: Sentence string.

        Returns:
            Resulting list of tokens.

        """
        return snt.split(" ")

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of filtered tokens

        """
        return snt

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of processed tokens.

        """
        return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        return snt

    def _tokenize(self, snt: str) -> Union[List[str], str]:
        """Tokenize text string.

        Args:
            snt: String.

        Returns:
            Resulting tokenized list.

        """
        res = self.preprocess_sentence(snt)
        res = self.tokenize_sentence(res)
        res = self.filter_tokens(res)
        res = self.postprocess_tokens(res)

        if self.to_string:
            res = " ".join(res)
            res = self.postprocess_sentence(res)
        return res

    def tokenize(self, text: List[str]) -> Union[List[List[str]], List[str]]:
        """Tokenize list of texts.

        Args:
            text: List of texts.

        Returns:
            Resulting tokenized list.

        """
        if self.n_jobs == 1:
            res = self._tokenize_singleproc(text)
        else:
            res = self._tokenize_multiproc(text)
        return res

    def _tokenize_singleproc(self, snt: List[str]) -> List[str]:
        """Singleproc version of tokenization.

        Args:
            snt: List of texts.

        Returns:
            List of tokenized texts.

        """
        return [self._tokenize(x) for x in snt]

    def _tokenize_multiproc(self, snt: List[str]) -> List[str]:
        """Multiproc version of tokenization.

        Args:
            snt: List of texts.

        Returns:
            List of tokenized texts.

        """
        idx = list(range(0, len(snt), len(snt) // self.n_jobs + 1)) + [len(snt)]

        parts = [snt[i:j] for (i, j) in zip(idx[:-1], idx[1:])]

        f = partial(tokenizer_func, tokenizer=self)
        with Pool(self.n_jobs) as p:
            res = p.map(f, parts)
        del f

        tokens = res[0]
        for r in res[1:]:
            tokens.extend(r)
        return tokens


class SimpleRuTokenizer(BaseTokenizer):
    """Russian tokenizer."""

    def __init__(
        self,
        n_jobs: int = 4,
        to_string: bool = True,
        stopwords: Optional[Union[bool, Sequence[str]]] = False,
        is_stemmer: bool = True,
        **kwargs: Any
    ):
        """Tokenizer for Russian language.

        Include numeric, punctuation and short word filtering.
        Use stemmer by default and do lowercase.

        Args:
            n_jobs: Number of threads for multiprocessing.
            to_string: Return string or list of tokens.
            stopwords: Use stopwords or not.
            is_stemmer: Use stemmer.

        """

        super().__init__(n_jobs, **kwargs)
        self.n_jobs = n_jobs
        self.to_string = to_string
        if isinstance(stopwords, (tuple, list, set)):
            self.stopwords = set(stopwords)
        elif stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words("russian"))
        else:
            self.stopwords = {}

        self.stemmer = SnowballStemmer("russian", ignore_stopwords=len(self.stopwords) > 0) if is_stemmer else None

    @staticmethod
    def _is_abbr(word: str) -> bool:
        """Check if the word is an abbreviation."""

        return sum([x.isupper() and x.isalpha() for x in word]) > 1 and len(word) <= 5

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        snt = snt.strip()
        snt = snt.replace("Ё", "Е").replace("ё", "е")
        s = re.sub("[^A-Za-zА-Яа-я0-9]+", " ", snt)
        s = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", s)
        return s

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: Sentence string.

        Returns:
            Resulting list of tokens.

        """
        return snt.split(" ")

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of filtered tokens.

        """

        filtered_s = []
        for w in snt:

            # ignore numbers
            if w.isdigit():
                pass
            elif w.lower() in self.stopwords:
                pass
            elif self._is_abbr(w):
                filtered_s.append(w)
            # ignore short words
            elif len(w) < 2:
                pass
            elif w.isalpha():
                filtered_s.append(w.lower())
        return filtered_s

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of processed tokens.

        """
        if self.stemmer is not None:
            return [self.stemmer.stem(w.lower()) for w in snt]
        else:
            return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        snt = (" " + snt).replace(" не ", " не")
        snt = snt.replace(" ни ", " ни")
        return snt[1:]


class SimpleEnTokenizer(BaseTokenizer):
    """English tokenizer."""

    def __init__(
        self,
        n_jobs: int = 4,
        to_string: bool = True,
        stopwords: Optional[Union[bool, Sequence[str]]] = False,
        is_stemmer: bool = True,
        **kwargs: Any
    ):
        """Tokenizer for English language.

        Args:
            n_jobs: Number of threads for multiprocessing.
            to_string: Return string or list of tokens.
            stopwords: Use stopwords or not.
            is_stemmer: Use stemmer.

        """

        super().__init__(n_jobs, **kwargs)
        self.n_jobs = n_jobs
        self.to_string = to_string
        if isinstance(stopwords, (tuple, list, set)):
            self.stopwords = set(stopwords)
        elif stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))
        else:
            self.stopwords = {}

        self.stemmer = SnowballStemmer("english", ignore_stopwords=len(self.stopwords) > 0) if is_stemmer else None

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        snt = snt.strip()
        s = re.sub("[^A-Za-zА-Яа-я0-9]+", " ", snt)
        s = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", s)
        return s

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: Sentence string.

        Returns:
            Resulting list of tokens.

        """
        return snt.split(" ")

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of filtered tokens.

        """
        if len(self.stopwords) > 0:
            filtered_s = []
            for w in snt:
                if w.lower() not in self.stopwords:
                    filtered_s.append(w)
            return filtered_s
        else:
            return snt

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of processed tokens.

        """
        if self.stemmer is not None:
            return [self.stemmer.stem(w.lower()) for w in snt]
        else:
            return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        return snt
