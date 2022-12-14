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

import cudf

from lightautoml_gpu.dataset.base import RolesDict
from lightautoml_gpu.dataset.roles import ColumnRole

Roles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]


class BaseTokenizerGPU:
    """Base class for tokenizer method.

    Tokenization with simple text cleaning and preprocessing.

    Args:
        to_string: Return string or list of tokens.
    """

    _fname_prefix = None
    _fit_checks = ()
    _transform_checks = ()

    def __init__(self, to_string: bool = True, **kwargs: Any):
        self.to_string = to_string
        self.stemmer = None

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
        return snt.str.split(" ")

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
        res = self.filter_tokens(res)
        res = self.postprocess_tokens(res)

        if self.to_string:
            if self.stemmer is not None:
                res = res.str.join(sep=" ")
            res = self.postprocess_sentence(res)
        return res

    def tokenize(self, text: List[str]) -> Union[List[List[str]], List[str]]:
        """Tokenize list of texts.

        Args:
            text: List of texts.

        Returns:
            Resulting tokenized list.

        """
        res = self._tokenize(text)
        return res


class SimpleRuTokenizerGPU(BaseTokenizerGPU):
    """Russian tokenizer.

    Tokenizer for Russian language.

    Include numeric, punctuation and short word filtering.
    Use stemmer by default and do lowercase.

    Args:
        to_string: Return string or list of tokens.
        stopwords: Use stopwords or not.
        is_stemmer: Use stemmer.
    """

    def __init__(
        self,
        to_string: bool = True,
        stopwords: Optional[Union[bool, Sequence[str]]] = False,
        is_stemmer: bool = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.to_string = to_string
        if isinstance(stopwords, (tuple, list, set)):
            self.stopwords = cudf.Series(set(stopwords))
        elif stopwords:
            self.stopwords = cudf.Series(set(nltk.corpus.stopwords.words("russian")))
        else:
            self.stopwords = cudf.Series()
        self.stemmer = SnowballStemmer("russian", ignore_stopwords=len(self.stopwords) > 0) if is_stemmer else None

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        s = snt.str.strip().str.lower()
        s = s.str.replace("ё", "е", regex=False)
        s = s.str.replace("[^A-Za-zА-Яа-я0-9]+", " ")
        s = s.str.filter_alphanum("  ")
        s = s.str.replace(r"^\d+\s|\s\d+\s|\s\d+$", " ")
        s = s.str.insert(0, " ").str.insert(-1, " ")
        s = s.str.replace(r"\s\d\w+\s|\s\w+\d\w+\s|\s\w+\d\s", " ")
        return s.str.normalize_spaces()

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: Sentence string.

        Returns:
            Resulting list of tokens.

        """
        return snt.str.split(" ")

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of filtered tokens.

        """
        # filter short tokens
        snt = snt.str.filter_tokens(min_token_length=2, replacement="")

        # filter stopwords
        if len(self.stopwords) > 0:
            snt = snt.str.replace_tokens(targets=self.stopwords, replacements="")
        return snt.str.normalize_spaces()

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of processed tokens.

        """
        if self.stemmer is not None:
            snt = snt.str.split(" ").to_pandas()
            for i in range(snt.shape[0]):
                snt[i] = [self.stemmer.stem(w) for w in snt[i]]
            snt = cudf.Series(snt)
        return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        snt = snt.str.insert(0, " ").str.replace(" не ", " не", regex=False)
        snt = snt.str.replace(" ни ", " ни", regex=False).str.strip()
        return snt


class SimpleEnTokenizerGPU(BaseTokenizerGPU):
    """English tokenizer.

    Tokenizer for English language.

    Args:
        to_string: Return string or list of tokens.
        stopwords: Use stopwords or not.
        is_stemmer: Use stemmer.
    """

    def __init__(
        self,
        to_string: bool = True,
        stopwords: Optional[Union[bool, Sequence[str]]] = False,
        is_stemmer: bool = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.to_string = to_string
        if isinstance(stopwords, (tuple, list, set)):
            self.stopwords = cudf.Series(set(stopwords))
        elif stopwords:
            self.stopwords = cudf.Series(set(nltk.corpus.stopwords.words("english")))
        else:
            self.stopwords = cudf.Series()

        self.stemmer = SnowballStemmer("english", ignore_stopwords=len(self.stopwords) > 0) if is_stemmer else None

    def preprocess_sentence(self, snt: str) -> str:
        """Preprocess sentence string (lowercase, etc.).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        snt = snt.str.strip()
        s = snt.str.replace("\n+", "")
        s = s.str.replace("[^A-Za-zА-Яа-я0-9]+", " ")
        s = s.str.replace(r"^\d+\s|\s\d+\s|\s\d+$", " ")
        return s.str.strip().str.lower()

    def tokenize_sentence(self, snt: str) -> List[str]:
        """Convert sentence string to a list of tokens.

        Args:
            snt: Sentence string.

        Returns:
            Resulting list of tokens.

        """
        return snt.str.split(" ")

    def filter_tokens(self, snt: List[str]) -> List[str]:
        """Clean list of sentence tokens.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of filtered tokens.

        """
        if len(self.stopwords) > 0:
            snt = snt.str.replace_tokens(targets=self.stopwords, replacements="")
        return snt.str.normalize_spaces()

    def postprocess_tokens(self, snt: List[str]) -> List[str]:
        """Additional processing steps: lemmatization, pos tagging, etc.

        Args:
            snt: List of tokens.

        Returns:
            Resulting list of processed tokens.

        """
        if self.stemmer is not None:
            snt = snt.str.split(" ").to_pandas()
            for i in range(len(snt)):
                snt[i] = [self.stemmer.stem(w) for w in snt[i]]
            snt = cudf.Series(snt)
        return snt

    def postprocess_sentence(self, snt: str) -> str:
        """Postprocess sentence string (merge words).

        Args:
            snt: Sentence string.

        Returns:
            Resulting string.

        """
        return snt
