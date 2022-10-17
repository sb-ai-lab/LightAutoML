"""Text features transformers."""

import gc
import logging
import os
import pickle

from copy import copy
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


try:
    import gensim
except:
    import warnings

    warnings.warn("'gensim' - package isn't installed")

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import CSRSparseDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import NumericRole
from ..dataset.roles import TextRole
from ..text.dl_transformers import BOREP
from ..text.dl_transformers import BertEmbedder
from ..text.dl_transformers import DLTransformer
from ..text.dl_transformers import RandomLSTM
from ..text.embed_dataset import BertDataset
from ..text.embed_dataset import EmbedDataset
from ..text.tokenizer import BaseTokenizer
from ..text.tokenizer import SimpleEnTokenizer
from ..text.utils import get_textarr_hash
from ..text.weighted_average_transformer import WeightedAverageTransformer
from .base import LAMLTransformer


logger = logging.getLogger(__name__)

NumpyOrPandas = Union[NumpyDataset, PandasDataset]
NumpyOrSparse = Union[NumpyDataset, CSRSparseDataset]

model_by_name = {
    "random_lstm": {
        "model": RandomLSTM,
        "model_params": {
            "embed_size": 300,
            "hidden_size": 256,
            "pooling": "mean",
            "num_layers": 1,
        },
        "dataset": EmbedDataset,
        "dataset_params": {
            "embedding_model": None,
            "max_length": 200,
            "embed_size": 300,
        },
        "loader_params": {"batch_size": 1024, "shuffle": False, "num_workers": 4},
        "embedding_model_params": {},
    },
    "random_lstm_bert": {
        "model": RandomLSTM,
        "model_params": {
            "embed_size": 768,
            "hidden_size": 256,
            "pooling": "mean",
            "num_layers": 1,
        },
        "dataset": BertDataset,
        "dataset_params": {"max_length": 256, "model_name": "bert-base-cased"},
        "loader_params": {"batch_size": 320, "shuffle": False, "num_workers": 4},
        "embedding_model": BertEmbedder,
        "embedding_model_params": {"model_name": "bert-base-cased", "pooling": "none"},
    },
    "borep": {
        "model": BOREP,
        "model_params": {
            "embed_size": 300,
            "proj_size": 300,
            "pooling": "mean",
            "max_length": 200,
            "init": "orthogonal",
            "pos_encoding": False,
        },
        "dataset": EmbedDataset,
        "dataset_params": {
            "embedding_model": None,
            "max_length": 200,
            "embed_size": 300,
        },
        "loader_params": {"batch_size": 1024, "shuffle": False, "num_workers": 4},
        "embedding_model_params": {},
    },
    "pooled_bert": {
        "model": BertEmbedder,
        "model_params": {"model_name": "bert-base-cased", "pooling": "mean"},
        "dataset": BertDataset,
        "dataset_params": {"max_length": 256, "model_name": "bert-base-cased"},
        "loader_params": {"batch_size": 320, "shuffle": False, "num_workers": 4},
        "embedding_model_params": {},
    },
    "wat": {
        "embedding_model": None,
        "embed_size": 300,
        "weight_type": "idf",
        "use_svd": True,
    },
}


def oof_task_check(dataset: LAMLDataset):
    """Check if task is binary or regression.

    Args:
        dataset: Dataset to check.

    """
    task = dataset.task
    assert task.name in [
        "binary",
        "reg",
    ], "Only binary and regression tasks supported in this transformer"


def text_check(dataset: LAMLDataset):
    """Check if all passed vars are text.

    Args:
        dataset: LAMLDataset to check.

    Raises:
         AssertionError: If non-text features are present.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[f].name == "Text", "Only text accepted in this transformer"


# TODO: combine TunableTransformer with LAMLTransformer class?
class TunableTransformer(LAMLTransformer):
    """Base class for ML transformers.

    Assume that parameters my set before training.
    """

    _default_params: dict = {}
    _params: dict = None

    @property
    def params(self) -> dict:
        """Parameters.

        Returns:
            Dict.

        """
        if self._params is None:
            self._params = copy(self.default_params)
        return self._params

    @params.setter
    def params(self, new_params: dict):
        assert isinstance(new_params, dict)
        self._params = {**self.params, **new_params}

    def init_params_on_input(self, dataset: NumpyOrPandas) -> dict:
        """Init params depending on input data.

        Returns:
            Dict with model hyperparameters.

        """
        return self.params

    def __init__(self, default_params: Optional[dict] = None, freeze_defaults: bool = True):
        """

        Args:
            default_params: algo hyperparams.
            freeze_defaults:
                - ``True`` :  params may be rewritten depending on dataset.
                - ``False``:  params may be changed only manually or with tuning.

        """
        self.task = None

        self.freeze_defaults = freeze_defaults
        if default_params is None:
            default_params = {}

        self.default_params = {**self._default_params, **default_params}


class TfidfTextTransformer(TunableTransformer):
    """Simple Tfidf vectorizer."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tfidf"
    _default_params = {
        "min_df": 5,
        "max_df": 1.0,
        "max_features": 30_000,
        "ngram_range": (1, 1),
        "analyzer": "word",
        "dtype": np.float32,
    }

    @property
    def features(self) -> List[str]:
        """Features list."""

        return self._features

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        subs: Optional[int] = None,
        random_state: int = 42,
    ):
        """

        Args:
            default_params: algo hyperparams.
            freeze_defaults: Flag.
            subs: Subsample to calculate freqs. If ``None`` - full data.
            random_state: Random state to take subsample.

        Note:
            The behaviour of `freeze_defaults`:

            - ``True`` :  params may be rewritten depending on dataset.
            - ``False``:  params may be changed only
              manually or with tuning.

        """
        super().__init__(default_params, freeze_defaults)
        self.subs = subs
        self.random_state = random_state
        self.vect = TfidfVectorizer
        self.dicts = {}

    def init_params_on_input(self, dataset: NumpyOrPandas) -> dict:
        """Get transformer parameters depending on dataset parameters.

        Args:
            dataset: Dataset used for model parmaeters initialization.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        suggested_params = copy(self.default_params)
        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params
        rows_num = len(dataset.data)
        if rows_num > 50_000:
            suggested_params["min_df"] = 25

        return suggested_params

    def fit(self, dataset: NumpyOrPandas):
        """Fit tfidf vectorizer.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        if self._params is None:
            self.params = self.init_params_on_input(dataset)
        dataset = dataset.to_pandas()
        df = dataset.data

        # fit ...
        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        feats = []
        for n, i in enumerate(subs.columns):
            vect = self.vect(**self.params)
            vect.fit(subs[i].fillna("").astype(str))
            features = list(
                np.char.array([self._fname_prefix + "_"])
                + np.arange(len(vect.vocabulary_)).astype(str)
                + np.char.array(["__" + i])
            )
            self.dicts[i] = {"vect": vect, "feats": features}
            feats.extend(features)
        self._features = feats
        return self

    def transform(self, dataset: NumpyOrPandas) -> CSRSparseDataset:
        """Transform text dataset to sparse tfidf representation.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Sparse dataset with encoded text.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        roles = NumericRole()
        outputs = []
        for n, i in enumerate(df.columns):
            new_arr = self.dicts[i]["vect"].transform(df[i].fillna("").astype(str))
            output = dataset.empty().to_numpy().to_csr()
            output.set_data(new_arr, self.dicts[i]["feats"], roles)
            outputs.append(output)
        # create resulted
        return dataset.empty().to_numpy().to_csr().concat(outputs)


class TokenizerTransformer(LAMLTransformer):
    """Simple tokenizer transformer."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tokenized"

    def __init__(self, tokenizer: BaseTokenizer = SimpleEnTokenizer()):
        """

        Args:
            tokenizer: text tokenizer.

        """
        self.tokenizer = tokenizer

    def transform(self, dataset: NumpyOrPandas) -> PandasDataset:
        """Transform text dataset to tokenized text dataset.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Pandas dataset with tokenized text.

        """

        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        roles = TextRole()
        outputs = []
        for n, i in enumerate(df.columns):
            pred = np.array(self.tokenizer.tokenize(df[i].fillna("").astype(str).tolist()))
            new_df = pd.DataFrame(pred, columns=[self._fname_prefix + "__" + i])
            outputs.append(new_df)
        # create resulted
        output = dataset.empty().to_pandas()
        output.set_data(pd.concat(outputs, axis=1), None, {feat: roles for feat in self.features})
        return output


class OneToOneTransformer(TunableTransformer):
    """Out-of-fold sgd model prediction to reduce dimension of encoded text data."""

    _fit_checks = (oof_task_check,)
    _transform_checks = ()
    _fname_prefix = "sgd_oof"
    _default_params = {"alpha": 0.0001, "max_iter": 1, "loss": "log"}

    @property
    def features(self) -> List[str]:
        """Features list."""

        return self._features

    def init_params_on_input(self, dataset: NumpyOrPandas) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            dataset: NumpyOrPandas.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        suggested_params = copy(self.default_params)

        self.task = dataset.task.name
        if self.task != "binary":
            suggested_params["loss"] = "squared_loss"

            algo = SGDRegressor
        else:
            algo = SGDClassifier

        self.algo = algo

        return suggested_params

    def __init__(self, default_params: Optional[int] = None, freeze_defaults: bool = False):
        super().__init__(default_params, freeze_defaults)
        """

        Args:
            default_params: Algo hyperparams.
            freeze_defaults:
                - ``True`` :  params may be rewritten depending on dataset.
                - ``False``:  params may be changed only manually or with tuning.
            subs: Subsample to calculate freqs. If None - full data.

        """

    def fit(self, dataset: NumpyOrPandas):
        """Apply fit transform.

        Args:
            dataset: Pandas or Numpy dataset of encoded text features.

        """
        for check_func in self._fit_checks:
            check_func(dataset)
        self.fit(dataset)

        for check_func in self._transform_checks:
            check_func(dataset)

        return self.transform(dataset)

    def fit_transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Fit and predict out-of-fold sgd model.

        Args:
            dataset: Pandas or Numpy dataset of encoded text features.

        Returns:
            Numpy dataset with out-of-fold model prediction.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)

        if self._params is None:
            self.params = self.init_params_on_input(dataset)
        dataset = dataset.to_numpy()
        data = dataset.data

        target = dataset.target.astype(np.int32)

        folds = dataset.folds
        n_folds = folds.max() + 1

        self.models = []

        oof_feats = np.zeros(len(data), dtype=np.float32)

        for n in range(n_folds):
            algo = self.algo(**self.params)
            algo.fit(data[folds != n], target[folds != n])
            if self.task == "binary":
                pred = algo.predict_proba(data[folds == n])[:, 1]
            else:
                pred = algo.predict(data[folds == n])
            oof_feats[folds == n] = pred

            self.models.append(deepcopy(algo))

        orig_name = dataset.features[0].split("__")[-1]
        self._features = [self._fname_prefix + "__" + orig_name]
        output = dataset.empty()
        self.output_role = NumericRole(np.float32, prob=output.task.name == "binary")
        output.set_data(oof_feats[:, np.newaxis], self.features, self.output_role)

        return output

    def transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Transform dataset to out-of-fold model-based encoding.

        Args:
            dataset: Pandas or Numpy dataset of encoded text features.

        Returns:
            Numpy dataset with out-of-fold model prediction.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        # transform
        out = np.zeros(len(data), dtype=np.float32)
        for n, model in enumerate(self.models):
            if self.task == "binary":
                pred = model.predict_proba(data)[:, 1]
            else:
                pred = model.predict(data)
            out += pred

        out /= len(self.models)
        # create resulted
        output = dataset.empty()
        output.set_data(out[:, np.newaxis], self.features, self.output_role)

        return output


class ConcatTextTransformer(LAMLTransformer):
    """Concat text features transformer."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "concated"

    def __init__(self, special_token: str = " [SEP] "):
        """

        Args:
            special_token: Add special token between columns.

        """
        self.special_token = special_token

    def transform(self, dataset: NumpyOrPandas) -> PandasDataset:
        """Transform text dataset to one text column.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Pandas dataset with one text column.

        """

        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        roles = TextRole()
        new_df = pd.DataFrame(
            df[df.columns].fillna("").astype(str).apply(f"{self.special_token}".join, axis=1),
            columns=[self._fname_prefix + "__" + "__".join(df.columns)],
        )
        # create resulted
        output = dataset.empty().to_pandas()
        output.set_data(new_df, None, {feat: roles for feat in new_df.columns})
        return output


class AutoNLPWrap(LAMLTransformer):
    """Calculate text embeddings."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "emb"
    fasttext_params = {"vector_size": 64, "window": 3, "min_count": 1}
    _names = {"random_lstm", "random_lstm_bert", "pooled_bert", "wat", "borep"}
    _trainable = {"wat", "borep", "random_lstm"}

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        model_name: str,
        embedding_model: Optional[str] = None,
        cache_dir: str = "./cache_NLP",
        bert_model: Optional[str] = None,
        transformer_params: Optional[Dict] = None,
        subs: Optional[int] = None,
        multigpu: bool = False,
        random_state: int = 42,
        train_fasttext: bool = False,
        fasttext_params: Optional[Dict] = None,
        fasttext_epochs: int = 2,
        sent_scaler: Optional[str] = None,
        verbose: bool = False,
        device: Any = "0",
        **kwargs: Any,
    ):
        """

        Args:
            model_name: Method for aggregating word embeddings
              into sentence embedding.
            transformer_params: Aggregating model parameters.
            embedding_model: Word level embedding model with dict
              interface or path to gensim fasttext model.
            cache_dir: If ``None`` - do not cache transformed datasets.
            bert_model: Name of HuggingFace transformer model.
            subs: Subsample to calculate freqs. If None - full data.
            multigpu: Use Data Parallel.
            random_state: Random state to take subsample.
            train_fasttext: Train fasttext.
            fasttext_params: Fasttext init params.
            fasttext_epochs: Number of epochs to train.
            verbose: Verbosity.
            device: Torch device or str.
            **kwargs: Unused params.

        """
        if train_fasttext:
            assert model_name in self._trainable, f"If train fasstext then model must be in {self._trainable}"

        assert model_name in self._names, f"Model name must be one of {self._names}"
        self.device = device
        self.multigpu = multigpu
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.subs = subs
        self.train_fasttext = train_fasttext
        self.fasttext_epochs = fasttext_epochs
        if fasttext_params is not None:
            self.fasttext_params.update(fasttext_params)
        self.dicts = {}
        self.sent_scaler = sent_scaler
        self.verbose = verbose
        self.model_name = model_name
        self.transformer_params = model_by_name[self.model_name]
        if transformer_params is not None:
            self.transformer_params.update(transformer_params)

        self._update_bert_model(bert_model)
        if embedding_model is not None:
            if isinstance(embedding_model, str):
                try:
                    embedding_model = gensim.models.FastText.load(embedding_model)
                except:
                    try:
                        embedding_model = gensim.models.FastText.load_fasttext_format(embedding_model)
                    except:
                        embedding_model = gensim.models.KeyedVectors.load(embedding_model)

            self.transformer_params = self._update_transformers_emb_model(self.transformer_params, embedding_model)

        else:

            self.train_fasttext = self.model_name in self._trainable

        if self.model_name == "wat":
            self.transformer = WeightedAverageTransformer
        else:
            self.transformer = DLTransformer

    def _update_bert_model(self, bert_model: str):
        if bert_model is not None:
            if "dataset_params" in self.transformer_params:
                self.transformer_params["dataset_params"]["model_name"] = bert_model
            if "embedding_model_params" in self.transformer_params:
                self.transformer_params["embedding_model_params"]["model_name"] = bert_model
            if "model_params" in self.transformer_params:
                self.transformer_params["model_params"]["model_name"] = bert_model
        return self

    def _update_transformers_emb_model(
        self, params: Dict, model: Any, emb_size: Optional[int] = None
    ) -> Dict[str, Any]:
        if emb_size is None:
            try:
                # Gensim checker [1]
                emb_size = model.vector_size
            except:
                try:
                    # Gensim checker[2]
                    emb_size = model.vw.vector_size
                except:
                    try:
                        # Natasha checker
                        emb_size = model[model.vocab.words[0]].shape[0]
                    except:
                        try:
                            # Dict of embeddings checker
                            emb_size = next(iter(model.values())).shape[0]
                        except:
                            raise ValueError("Unrecognized embedding dimention, please specify it in model_params")
        try:
            model = model.wv
        except:
            pass

        if self.model_name == "wat":
            params["embed_size"] = emb_size
            params["embedding_model"] = model
        elif self.model_name in {"random_lstm", "borep"}:
            params["dataset_params"]["embedding_model"] = model
            params["dataset_params"]["embed_size"] = emb_size
            params["model_params"]["embed_size"] = emb_size

        return params

    def fit(self, dataset: NumpyOrPandas):
        """Fit chosen transformer and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        """
        for check_func in self._fit_checks:
            check_func(dataset)

        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # fit ...
        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        names = []
        for n, i in enumerate(subs.columns):
            transformer_params = deepcopy(self.transformer_params)
            if self.train_fasttext:
                embedding_model = gensim.models.FastText(**self.fasttext_params)
                common_texts = [i.split(" ") for i in subs[i].values]
                embedding_model.build_vocab(corpus_iterable=common_texts)
                embedding_model.train(
                    corpus_iterable=common_texts,
                    total_examples=len(common_texts),
                    epochs=self.fasttext_epochs,
                )
                transformer_params = self._update_transformers_emb_model(transformer_params, embedding_model)

            transformer = self.transformer(
                verbose=self.verbose,
                device=self.device,
                multigpu=self.multigpu,
                **transformer_params,
            )
            emb_name = transformer.get_name()
            emb_size = transformer.get_out_shape()

            feats = [self._fname_prefix + "_" + emb_name + "_" + str(x) + "__" + i for x in range(emb_size)]

            self.dicts[i] = {
                "transformer": deepcopy(transformer.fit(subs[i])),
                "feats": feats,
            }
            names.extend(feats)
            logger.info3(f"Feature {i} fitted")

            del transformer
            gc.collect()
            torch.cuda.empty_cache()

        self._features = names
        return self

    def transform(self, dataset: NumpyOrPandas) -> NumpyOrPandas:
        """Transform tokenized dataset to text embeddings.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Numpy dataset with text embeddings.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        roles = NumericRole()
        outputs = []
        for n, i in enumerate(df.columns):
            if self.cache_dir is not None:
                full_hash = get_textarr_hash(df[i]) + get_textarr_hash(self.dicts[i]["feats"])
                fname = os.path.join(self.cache_dir, full_hash + ".pkl")
                if os.path.exists(fname):
                    logger.info3(f"Load saved dataset for {i}")
                    with open(fname, "rb") as f:
                        new_arr = pickle.load(f)
                else:
                    new_arr = self.dicts[i]["transformer"].transform(df[i])
                    with open(fname, "wb") as f:
                        pickle.dump(new_arr, f)
            else:
                new_arr = self.dicts[i]["transformer"].transform(df[i])

            output = dataset.empty().to_numpy()
            output.set_data(new_arr, self.dicts[i]["feats"], roles)
            outputs.append(output)
            logger.info3(f"Feature {i} transformed")
        # create resulted
        dataset = dataset.empty().to_numpy().concat(outputs)
        # instance-wise sentence embedding normalization
        dataset.data = dataset.data / self._sentence_norm(dataset.data, self.sent_scaler)

        return dataset

    @staticmethod
    def _sentence_norm(x: np.ndarray, mode: Optional[str] = None) -> Union[np.ndarray, float]:
        """Get sentence embedding norm."""
        if mode == "l2":
            return ((x ** 2).sum(axis=1, keepdims=True)) ** 0.5
        elif mode == "l1":
            return np.abs(x).sum(axis=1, keepdims=True)
        if mode is not None:
            logger.info2("Unknown sentence scaler mode: sent_scaler={}, " "no normalization will be used".format(mode))
        return 1
