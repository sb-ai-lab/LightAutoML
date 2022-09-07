"""Text features."""

from typing import Any

from ...dataset.base import LAMLDataset
from ...text.tokenizer import BaseTokenizer
from ...text.tokenizer import SimpleEnTokenizer
from ...text.tokenizer import SimpleRuTokenizer
from ...transformers.base import ColumnsSelector
from ...transformers.base import LAMLTransformer
from ...transformers.base import SequentialTransformer
from ...transformers.base import UnionTransformer
from ...transformers.decomposition import SVDTransformer
from ...transformers.numeric import StandardScaler
from ...transformers.text import AutoNLPWrap
from ...transformers.text import ConcatTextTransformer
from ...transformers.text import TfidfTextTransformer
from ...transformers.text import TokenizerTransformer
from ..utils import get_columns_by_role
from .base import FeaturesPipeline


_model_name_by_lang = {
    "ru": "DeepPavlov/rubert-base-cased-conversational",  # "sberbank-ai/sbert_large_nlu_ru" - sberdevices
    "en": "bert-base-cased",
    "multi": "bert-base-multilingual-cased",
}

_tokenizer_by_lang = {
    "ru": SimpleRuTokenizer,
    "en": SimpleEnTokenizer,
    "multi": BaseTokenizer,
}


class NLPDataFeatures:
    """
    Class contains basic features transformations for text data.
    """

    _lang = {"en", "ru", "multi"}

    def __init__(self, **kwargs: Any):
        """Set default parameters for nlp pipeline constructor.

        Args:
            **kwargs: default params.

        """
        if "lang" in kwargs:
            assert kwargs["lang"] in self._lang, f"Language must be one of: {self._lang}"

        self.lang = "en" if "lang" not in kwargs else kwargs["lang"]
        self.is_tokenize_autonlp = False
        self.use_stem = False
        self.verbose = False
        self.bert_model = _model_name_by_lang[self.lang]
        self.random_state = 42
        self.device = None
        self.model_name = None
        self.embedding_model = None
        self.svd = True
        self.n_components = 100
        self.is_concat = True
        self.tfidf_params = None
        self.cache_dir = None
        self.train_fasttext = False
        self.embedding_model = None  # path to fasttext model or model with dict interface
        self.transformer_params = None  # params of random_lstm, bert_embedder, borep or wat
        self.fasttext_params = None  # init fasttext params
        self.fasttext_epochs = 2
        self.stopwords = False
        self.force = False
        self.sent_scaler = None
        self.embed_scaler = None
        # if in autonlp_params no effect
        self.multigpu = False

        for k in kwargs:
            if kwargs[k] is not None:
                self.__dict__[k] = kwargs[k]

        if not self.force and self.device == "cpu":
            self.model_name = "wat"
        else:
            if self.model_name is None:
                self.model_name = (
                    "wat"
                    if self.device == "cpu"
                    else "random_lstm"
                    if "embedding_model" in kwargs
                    else "random_lstm_bert"
                )


class TextAutoFeatures(FeaturesPipeline, NLPDataFeatures):
    """
    Class contains embedding features for text data.
    """

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        """Create pipeline for textual data.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """
        transformers_list = []
        # process texts
        texts = get_columns_by_role(train, "Text")
        if len(texts) > 0:
            transforms = [ColumnsSelector(keys=texts)]
            if self.is_concat:
                transforms.append(ConcatTextTransformer())
            if self.is_tokenize_autonlp:
                transforms.append(
                    TokenizerTransformer(
                        tokenizer=_tokenizer_by_lang[self.lang](is_stemmer=self.use_stem, stopwords=self.stopwords)
                    )
                )
            transforms.append(
                AutoNLPWrap(
                    model_name=self.model_name,
                    embedding_model=self.embedding_model,
                    cache_dir=self.cache_dir,
                    bert_model=self.bert_model,
                    transformer_params=self.transformer_params,
                    random_state=self.random_state,
                    train_fasttext=self.train_fasttext,
                    device=self.device,
                    multigpu=self.multigpu,
                    sent_scaler=self.sent_scaler,
                    fasttext_params=self.fasttext_params,
                    fasttext_epochs=self.fasttext_epochs,
                    verbose=self.verbose,
                )
            )
            if self.embed_scaler == "standard":
                transforms.append(StandardScaler())

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class NLPTFiDFFeatures(FeaturesPipeline, NLPDataFeatures):
    """
    Class contains tfidf features for text data.
    """

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        """Create pipeline for textual data.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """
        transformers_list = []

        # process texts
        texts = get_columns_by_role(train, "Text")
        if len(texts) > 0:
            transforms = [
                ColumnsSelector(keys=texts),
                TokenizerTransformer(
                    tokenizer=_tokenizer_by_lang[self.lang](is_stemmer=self.use_stem, stopwords=self.stopwords)
                ),
                TfidfTextTransformer(default_params=self.tfidf_params, subs=None, random_state=42),
            ]
            if self.svd:
                transforms.append(SVDTransformer(n_components=self.n_components))

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class TextBertFeatures(FeaturesPipeline, NLPDataFeatures):
    """
    Features pipeline for BERT.
    """

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        """Create pipeline for BERT.

        Args:
            train: Dataset with train data.

        Returns:
            Transformer.

        """
        transformers_list = []

        # process texts
        texts = get_columns_by_role(train, "Text")
        if len(texts) > 0:
            text_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=texts),
                    ConcatTextTransformer(),
                ]
            )
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all
