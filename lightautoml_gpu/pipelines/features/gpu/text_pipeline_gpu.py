"""Text features."""

from typing import Any

from lightautoml_gpu.dataset.base import LAMLDataset
from lightautoml_gpu.text.gpu.tokenizer_gpu import BaseTokenizerGPU
from lightautoml_gpu.text.gpu.tokenizer_gpu import SimpleEnTokenizerGPU
from lightautoml_gpu.text.gpu.tokenizer_gpu import SimpleRuTokenizerGPU
from lightautoml_gpu.transformers.base import ColumnsSelector
from lightautoml_gpu.transformers.base import LAMLTransformer
from lightautoml_gpu.transformers.base import SequentialTransformer
from lightautoml_gpu.transformers.base import UnionTransformer
from lightautoml_gpu.transformers.gpu.numeric_gpu import StandardScalerGPU
from lightautoml_gpu.transformers.gpu.text_gpu import AutoNLPWrapGPU
from lightautoml_gpu.transformers.gpu.text_gpu import ConcatTextTransformerGPU
from lightautoml_gpu.transformers.gpu.text_gpu import TfidfTextTransformerGPU
from lightautoml_gpu.transformers.gpu.text_gpu import TokenizerTransformerGPU
from lightautoml_gpu.transformers.gpu.text_gpu import SubwordTokenizerTransformerGPU
from lightautoml_gpu.pipelines.utils import get_columns_by_role
from lightautoml_gpu.pipelines.features.base import FeaturesPipeline

from lightautoml_gpu.pipelines.features.text_pipeline import _model_name_by_lang

_tokenizer_by_lang = {
    "ru": SimpleRuTokenizerGPU,
    "en": SimpleEnTokenizerGPU,
    "multi": BaseTokenizerGPU,
}


class NLPDataFeaturesGPU:
    """
    Class contains basic features transformations for text data.

    Set default parameters for nlp pipeline constructor.

    Args:
        **kwargs: default params.
    """

    _lang = {"en", "ru", "multi"}

    def __init__(self, **kwargs: Any):
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
        self.word_vectors_cache = None
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
        # subword parameters
        self.vocab_path = None
        self.data_path = None
        self.is_hash = False
        self.max_length = 300
        self.tokenizer = "fasttext"
        self.vocab_size = 30000
        self.tokenizer_save_path = f"{self.tokenizer}_vocab_{self.vocab_size//1000}k.txt"

        for k in kwargs:
            if kwargs[k] is not None:
                self.__dict__[k] = kwargs[k]

        if self.model_name is None:
            self.model_name = (
                "random_lstm"
                if "embedding_model" in kwargs
                else "random_lstm_bert"
            )


class TextAutoFeaturesGPU(FeaturesPipeline, NLPDataFeaturesGPU):
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
                transforms.append(ConcatTextTransformerGPU())
            if self.is_tokenize_autonlp:
                transforms.append(
                    TokenizerTransformerGPU(
                        tokenizer=_tokenizer_by_lang[self.lang](is_stemmer=self.use_stem, stopwords=self.stopwords)
                    )
                )
            transforms.append(
                AutoNLPWrapGPU(
                    model_name=self.model_name,
                    embedding_model=self.embedding_model,
                    word_vectors_cache=self.word_vectors_cache,
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
                    lang=self.lang
                )
            )
            if self.embed_scaler == "standard":
                transforms.append(StandardScalerGPU())

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class NLPTFiDFFeaturesGPU(FeaturesPipeline, NLPDataFeaturesGPU):
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
                TokenizerTransformerGPU(
                    tokenizer=_tokenizer_by_lang[self.lang](is_stemmer=self.use_stem, stopwords=self.stopwords)
                ),
                TfidfTextTransformerGPU(default_params=self.tfidf_params, subs=None, random_state=42, n_components=self.n_components),
            ]

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class NLPTFiDFFeaturesSubwordGPU(FeaturesPipeline, NLPDataFeaturesGPU):
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
                SubwordTokenizerTransformerGPU(vocab_path=self.vocab_path, data_path=self.data_path,
                                               is_hash=self.is_hash,
                                               max_length=self.max_length, tokenizer=self.tokenizer,
                                               vocab_size=self.vocab_size, save_path=self.tokenizer_save_path),
                TfidfTextTransformerGPU(default_params=self.tfidf_params, subs=None, random_state=42,
                                        n_components=self.n_components),
            ]

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class TextBertFeaturesGPU(FeaturesPipeline, NLPDataFeaturesGPU):
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
                    ConcatTextTransformerGPU(),
                ]
            )
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all
