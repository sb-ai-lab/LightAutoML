"""Text features transformers."""

import codecs
import gc
import logging
import os
import pickle
import time

from copy import copy
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple

import numpy as np
import cupy as cp
import pandas as pd
import cudf
import dask_cudf
import torch
import cupyx
from cupyx.scipy.sparse.linalg import svds
from cudf.core.subword_tokenizer import SubwordTokenizer
from cudf.utils.hash_vocab_utils import hash_vocab
import dask.array as da

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

import torchnlp
from torchnlp.word_to_vector import BPEmb
from torchnlp.word_to_vector import FastText

from cuml.feature_extraction.text import TfidfVectorizer

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.gpu.gpu_dataset import CupySparseDataset
from lightautoml.dataset.roles import TextRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.text.gpu.tokenizer_gpu import BaseTokenizerGPU
from lightautoml.text.gpu.tokenizer_gpu import SimpleEnTokenizerGPU
from lightautoml.text.gpu.tokenizer_gpu import SimpleRuTokenizerGPU
from lightautoml.text.gpu.dl_transformers_gpu import BOREPGPU
from lightautoml.text.gpu.dl_transformers_gpu import BertEmbedderGPU
from lightautoml.text.gpu.dl_transformers_gpu import DLTransformerGPU
from lightautoml.text.gpu.dl_transformers_gpu import RandomLSTMGPU
from lightautoml.text.gpu.embed_dataset_gpu import EmbedDatasetGPU
from lightautoml.text.gpu.embed_dataset_gpu import BertDatasetGPU
from lightautoml.text.utils import get_textarr_hash
from lightautoml.text.gpu.weighted_average_transformer_gpu import WeightedAverageTransformerGPU

from lightautoml.transformers.gpu.svd_utils_gpu import _svd_lowrank

from lightautoml.transformers.text import oof_task_check
from lightautoml.transformers.text import text_check
from lightautoml.transformers.text import TunableTransformer

logger = logging.getLogger(__name__)

GpuNumericalDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]

model_by_name = {
    "random_lstm": {
        "model": RandomLSTMGPU,
        "model_params": {
            "embed_size": 300,
            "hidden_size": 256,
            "pooling": "mean",
            "num_layers": 1,
        },
        "dataset": EmbedDatasetGPU,
        "dataset_params": {
            "embedding_model": None,
            "max_length": 200,
            "embed_size": 300,
        },
        "loader_params": {"batch_size": 1024, "shuffle": False, "num_workers": 1},
        "embedding_model_params": {},
    },
    "random_lstm_bert": {
        "model": RandomLSTMGPU,
        "model_params": {
            "embed_size": 768,
            "hidden_size": 256,
            "pooling": "mean",
            "num_layers": 1,
        },
        "dataset": BertDatasetGPU,
        "dataset_params": {"max_length": 256, "model_name": "bert-base-cased"},
        "loader_params": {"batch_size": 320, "shuffle": False, "num_workers": 1},
        "embedding_model": BertEmbedderGPU,
        "embedding_model_params": {"model_name": "bert-base-cased", "pooling": "none"},
    },
    "borep": {
        "model": BOREPGPU,
        "model_params": {
            "embed_size": 300,
            "proj_size": 300,
            "pooling": "mean",
            "max_length": 200,
            "init": "orthogonal",
            "pos_encoding": False,
        },
        "dataset": EmbedDatasetGPU,
        "dataset_params": {
            "embedding_model": None,
            "max_length": 200,
            "embed_size": 300,
        },
        "loader_params": {"batch_size": 1024, "shuffle": False, "num_workers": 1},
        "embedding_model_params": {},
    },
    "pooled_bert": {
        "model": BertEmbedderGPU,
        "model_params": {"model_name": "bert-base-cased", "pooling": "mean"},
        "dataset": BertDatasetGPU,
        "dataset_params": {"max_length": 256, "model_name": "bert-base-cased"},
        "loader_params": {"batch_size": 320, "shuffle": False, "num_workers": 1},
        "embedding_model_params": {},
    },
    "wat": {
        "embedding_model": None,
        "embed_size": 300,
        "weight_type": "idf",
        "use_svd": True,
    },
}


class TunableTransformerGPU(LAMLTransformer):
    """Base class for ML transformers (GPU).

    Assume that parameters my set before training.

    Args:
        default_params: algo hyperparams.
        freeze_defaults:
            - ``True`` :  params may be rewritten depending on dataset.
            - ``False``:  params may be changed only manually or with tuning.
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

    def init_params_on_input(self, dataset: GpuNumericalDataset) -> dict:
        """Init params depending on input data.

        Returns:
            Dict with model hyperparameters.

        """
        return self.params

    def __init__(self, default_params: Optional[dict] = None, freeze_defaults: bool = True):
        self.task = None

        self.freeze_defaults = freeze_defaults
        if default_params is None:
            default_params = {}

        self.default_params = {**self._default_params, **default_params}


class TfidfTextTransformerGPU(TunableTransformerGPU):
    """Simple Tfidf vectorizer followed by SVD (GPU).

    Args:
        default_params: algo hyperparams.
        freeze_defaults: Flag.
        subs: Subsample to calculate freqs. If ``None`` - full data.
        random_state: Random state to take subsample.
        n_components: Number of components in TruncatedSVD
        n_oversample: Number os oversample components in TruncatedSVD
        n_iter: Number of iterations in SVD algorithm

    Note:
        The behaviour of `freeze_defaults`:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only
          manually or with tuning.
    """

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tfidfGPU"
    _default_params = {
        "min_df": 5, 
        "max_df": 1.0,
        "max_features": 30_000,
        "ngram_range": (1, 1),
        "analyzer": "word",
        "dtype": cp.float32,
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
        n_components: int = 100,
        n_oversample: int = 0,
        n_iter: int = 2,
    ):
        super().__init__(default_params, freeze_defaults)
        self.subs = subs
        self.random_state = random_state
        # cuml tf-idf vectorizer for one gpu
        self.vect = TfidfVectorizer
        self.vocab_len = 0

        self.dicts = {}
        # variables for svd
        self.n_components = n_components
        self.n_oversample = n_oversample
        self.n_iter = n_iter
        self.Vh = None

    def init_params_on_input(self, dataset: GpuNumericalDataset) -> dict:
        """Get transformer parameters depending on dataset parameters.

        Args:
            dataset: Dataset used for model parameters initialization.

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

    # convert csr sparse matrix to coo format
    def _csr2coo(self, arr):
        coo = arr.tocoo()
        i = torch.as_tensor(cp.vstack((coo.row, coo.col)), dtype=torch.int64, device='cuda')
        v = torch.as_tensor(coo.data, dtype=torch.float32, device='cuda')
        coo = torch.cuda.sparse.FloatTensor(i, v, torch.Size(coo.shape))
        return coo

    # calculate tfidf (followed by SVD if flag 'svd=True')
    def _calc_tfidf(self, df, svd=False):
        outputs = []
        for col in df.columns:
            new_arr = self.dicts[col]["vect"].transform(df[col].fillna("").astype(str))
            outputs.append(new_arr)
        if svd:
            return cupyx.scipy.sparse.hstack(outputs, format='csr') @ self.Vh
        else:
            return cupyx.scipy.sparse.hstack(outputs, format='csr')

    def _fit_cupy(self, dataset: GpuNumericalDataset):
        dataset = dataset.to_cudf()
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

        outputs = []
        for n, i in enumerate(df.columns):
            output = self.dicts[i]["vect"].transform(df[i].fillna("").astype(str))
            outputs.append(output)
        output = self._csr2coo(cupyx.scipy.sparse.hstack(outputs, format='csr'))

        _, _, Vh = _svd_lowrank(output, q=self.n_components+self.n_oversample, niter=self.n_iter)
        self.Vh = cp.array(Vh[:, :self.n_components])
        return self

    def _fit_daskcudf(self, dataset: GpuNumericalDataset):
        df = dataset.data

        # fit ...
        if self.subs is not None and len(df) >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        feats = []
        for n, i in enumerate(subs.columns):
            vect = self.vect(**self.params)
            vect.fit(subs[i].fillna("").astype(str).compute())
            features = list(
                np.char.array([self._fname_prefix + "_"])
                + np.arange(len(vect.vocabulary_)).astype(str)
                + np.char.array(["__" + i])
            )
            self.dicts[i] = {"vect": vect, "feats": features}
            feats.extend(features)
        self._features = feats

        output = df.map_partitions(self._calc_tfidf, svd=False,
                                   meta=cupyx.scipy.sparse.csr_matrix(
                                       (len(df.partitions[0]), self._default_params["max_features"]))).persist()

        rs = da.random.RandomState(RandomState=cp.random.RandomState)
        _, _, Vh = da.linalg.svd_compressed(output, k=self.n_components, seed=rs,
                                            n_oversamples=self.n_oversample)

        self.Vh = cupyx.scipy.sparse.csr_matrix(Vh.T.compute())
        return self

    def fit(self, dataset: GpuNumericalDataset):
        """Fit tfidf vectorizer (GPU version).

        Args:
            dataset: Cudf or Daskcudf dataset of text features.

        Returns:
            self.

        """
        for check_func in self._fit_checks:
            check_func(dataset)

        if self._params is None:
            self.params = self.init_params_on_input(dataset)

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        if isinstance(dataset, DaskCudfDataset):
            self._fit_daskcudf(dataset)
        else:
            self._fit_cupy(dataset)
        return self

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        roles = NumericRole(cp.float32)

        new_arr = self._calc_tfidf(df, svd=True)

        output = dataset.empty().to_cupy()
        output.set_data(new_arr, None, roles)
        return output

    def _transform_daskcudf(self, dataset: GpuNumericalDataset) -> DaskCudfDataset:
        df = dataset.data
        gpu_cnt = df.npartitions

        # transform
        roles = NumericRole(cp.float32)

        new_arr = df.map_partitions(self._calc_tfidf, svd=True,
                                    meta=cp.empty((len(df.partitions[0]), self.n_components))).compute().toarray()
        new_arr = cudf.DataFrame(new_arr)
        new_arr = dask_cudf.from_cudf(new_arr, npartitions=gpu_cnt)

        output = dataset.empty()
        output.set_data(new_arr, None, {feat: roles for feat in self.features})

        # create resulted
        return output

    def transform(self, dataset: GpuNumericalDataset) -> CupyDataset:
        """Transform text dataset to sparse tfidf representation which is made dense by SVD transform.

        Args:
            dataset: Cudf or DaskCudf dataset of text features.

        Returns:
            For Cudf dataset it returns Cupy dataset obtained by tfidf transformer and SVD.
            For Daskcudf dataset it returns Daskcudf dataset obtained by tfidf transformer and SVD.

        """

        # checks here
        super().transform(dataset)

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)

    def _fit_transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        # fit ...
        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        # fit Vectorizer on a (potentially) subset of df
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

        # transform df with Vectorizer
        outputs = []
        for n, i in enumerate(df.columns):
            output = self.dicts[i]["vect"].transform(df[i].fillna("").astype(str))
            outputs.append(output)
        output = cupyx.scipy.sparse.hstack(outputs, format='csr')

        # calculate SVD
        _, _, Vh = _svd_lowrank(self._csr2coo(output), q=self.n_components+self.n_oversample, niter=self.n_iter)
        self.Vh = cp.array(Vh[:, :self.n_components])

        # apply SVD
        self._features = [idx for idx in range(self.n_components)]
        roles = NumericRole(cp.float32)
        new_arr = output @ self.Vh

        output = dataset.empty().to_cupy()
        output.set_data(new_arr, None, roles)
        return output

    def _fit_transform_daskcudf(self, dataset: GpuNumericalDataset) -> DaskCudfDataset:
        df = dataset.data
        gpu_cnt = torch.cuda.device_count()

        # fit ...
        if self.subs is not None and len(df) >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df.copy()

        # fit Vectorizer on a single GPU
        feats = []
        for n, i in enumerate(subs.columns):
            vect = self.vect(**self.params)
            vect.fit(subs[i].fillna("").astype(str).compute())
            self.vocab_len += len(vect.vocabulary_)
            features = list(
                np.char.array([self._fname_prefix + "_"])
                + np.arange(len(vect.vocabulary_)).astype(str)
                + np.char.array(["__" + i])
            )
            self.dicts[i] = {"vect": vect, "feats": features}
            feats.extend(features)
        self._features = feats

        # transform df with Vectorizer on multiple-GPU(?)
        output = df.map_partitions(self._calc_tfidf, svd=False,
                                   meta=cupyx.scipy.sparse.csr_matrix(
                                       (len(df.partitions[0]), self.vocab_len))).persist()

        rs = da.random.RandomState(RandomState=cp.random.RandomState)
        _, _, Vh = da.linalg.svd_compressed(output, k=self.n_components, seed=rs)

        self.Vh = cupyx.scipy.sparse.csr_matrix(Vh.T.compute())

        new_arr = output.map_blocks(lambda x: (x @ self.Vh).toarray(), dtype=cp.float32).compute()
        new_arr = cudf.DataFrame(new_arr)
        new_arr = dask_cudf.from_cudf(new_arr, npartitions=gpu_cnt)

        roles = NumericRole(cp.float32)
        self._features = [idx for idx in range(self.n_components)]

        output = dataset.empty()
        output.set_data(new_arr, None, {feat: roles for feat in self.features})
        return output

    def fit_transform(self, dataset: GpuNumericalDataset) -> CupySparseDataset:
        """Fit and transform text dataset to sparse tfidf representation which is made dense by SVD transform.
        This 'fit_transform' method is much more efficient than 'fit' followed by 'transform' method.

        Args:
            dataset: Cudf or DaskCudf dataset of text features.

        Returns:
            For Cudf dataset it returns Cupy dataset obtained by tfidf transformer and SVD.
            For Daskcudf dataset it returns Daskcudf dataset obtained by tfidf transformer and SVD.

        """

        # checks here
        for check_func in self._fit_checks:
            check_func(dataset)

        if self._params is None:
            self.params = self.init_params_on_input(dataset)
        super().transform(dataset)

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        if isinstance(dataset, DaskCudfDataset):
            return self._fit_transform_daskcudf(dataset)
        else:
            return self._fit_transform_cupy(dataset)


class TokenizerTransformerGPU(LAMLTransformer):
    """Simple tokenizer transformer (GPU).

    Args:
        tokenizer: text tokenizer.
    """

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tokenizedGPU"

    def __init__(self, tokenizer: BaseTokenizerGPU = SimpleEnTokenizerGPU()):
        self.tokenizer = tokenizer

    # apply tokenization to each text column
    def _tokenize_columns(self, df):
        outputs = []
        for col in df.columns:
            pred = self.tokenizer.tokenize(df[col].fillna("").astype(str))
            new_df = pred.rename(self._fname_prefix + "__" + col)
            outputs.append(new_df)
        new_df = cudf.DataFrame(cudf.concat(outputs, axis=1))
        return new_df

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> CudfDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        roles = TextRole()
        new_df = self._tokenize_columns(df)

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        df = dataset.data
        
        roles = TextRole()
        new_cols = [self._fname_prefix + "__" + col for col in df.columns]

        # Since stemmer works on CPU, here is an unpretty workaround
        if self.tokenizer.stemmer is None:
            new_df = df.map_partitions(self._tokenize_columns, meta=cudf.DataFrame(columns=new_cols).astype(str)).persist()
        else:
            gpu_cnt = torch.cuda.device_count()
            outputs, new_cols = [], []
            for i, col in enumerate(df.columns):
                pred = self.tokenizer.tokenize(df[col].compute().fillna("").astype(str))
                new_col_name = self._fname_prefix + "__" + col
                new_cols.append(new_col_name)
                new_df = pred.rename(new_col_name)
                new_df = dask_cudf.from_cudf(new_df, npartitions=gpu_cnt)
                outputs.append(new_df)
            col_map = dict(zip(np.arange(len(df.columns)), new_cols))
            new_df = dask_cudf.concat(outputs, axis=1).rename(columns=col_map)
        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform text dataset to tokenized text dataset.

        Args:
            dataset: Cudf or DaskCudf dataset of text features.

        Returns:
            Respective dataset with tokenized text.

        """

        # checks here
        super().transform(dataset)
        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class SubwordTokenizerTransformerGPU(LAMLTransformer):
    """Subword tokenizer transformer (GPU).

    Args:
        vocab_path: path to vocabulary .txt file,
        data_path: .txt file (saved pd.Series) for the tokenizer to be trained on (if vocab is not specified)
        is_hash: True means vocab is not raw vocab but was transformed with hash_vocab function from cudf,
        max_length: max number of tokens to leave in one text (exceeding ones would be truncated)
        tokenizer: ["bpe" or "wordpiece"] if vocab is None. Type of tokenizer to be trained
        vocab_size: vocabulary size for trained tokenizer
        save_path: path where trained vocabulary would be saved to
    """

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "subword_tokenizedGPU"

    def __init__(self, vocab_path: str = None, data_path: Any = None, is_hash: bool = False, max_length: int = 300,
                 tokenizer: str = "bpe", vocab_size: int = 30000,
                 save_path: str = None):
        assert vocab_path is not None or data_path is not None, "To use subword tokenizer you need to pass either " \
                                                      "path to tokenizer vocabulary or path to string data to be trained on"
        if vocab_path is None:
            assert tokenizer in ["bpe", "wordpiece"], f"Only bpe and wordpiece tokenizers are available for train" \
                                                      f"but you passed {tokenizer}"
            save_path = save_path if save_path is not None else f"{tokenizer}_vocab_{vocab_size // 1000}k.txt"
            if tokenizer == "bpe":
                tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
                trainer = BpeTrainer(
                    vocab_size=vocab_size, special_tokens=["[UNK]", "[SEP]", "[CLS]"]
                )
            else:
                tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
                trainer = WordPieceTrainer(
                    vocab_size=vocab_size, special_tokens=["[UNK]", "[SEP]", "[CLS]"]
                )
            tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
            tokenizer.pre_tokenizer = Whitespace()

            tokenizer.train([data_path], trainer)
            trained_vocab = tokenizer.get_vocab()

            with codecs.open(save_path, 'w+', 'utf-8') as f:
                for key in trained_vocab.keys():
                    f.write(key + '\n')

            vocab_path = save_path.split(".")[0] + '_hash.txt'
            hash_vocab(save_path, vocab_path)
        elif not is_hash:
            hash_vocab(vocab_path, vocab_path.split('.')[0]+'_hash.txt')
            vocab_path = vocab_path.split('.')[0]+'_hash.txt'
        self.vocab_path = vocab_path
        self.max_length = max_length
        # SOS, EOS, padding tokens to filter
        self.targets = ['101', '102', '0']

    # concatenate text columns
    @staticmethod
    def _concat_columns(df, col_name):
        outputs = df[df.columns[0]].fillna("").astype(str)
        for col in df.columns[1:]:
            outputs = outputs + " " + df[col].fillna("").astype(str)
        new_df = cudf.DataFrame({col_name: outputs})
        return new_df

    # subword tokenize text columns
    def _subword_tokenize_columns(self, df):
        outputs = []
        subword_gpu = SubwordTokenizer(self.vocab_path)
        for col in df.columns:
            pred = subword_gpu(df[col].fillna("").astype(str),
                               max_length=self.max_length,
                               max_num_rows=len(df[col]),
                               padding='max_length',
                               return_tensors='cp',
                               truncation=True)
            col_name = self._fname_prefix + "__" + col
            pred = self._concat_columns(cudf.DataFrame(pred['input_ids']).astype(str), col_name)
            pred = pred[col_name].str.replace_tokens(self.targets, "").str.normalize_spaces()
            outputs.append(pred)
        new_df = cudf.DataFrame(cudf.concat(outputs, axis=1))
        return new_df

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> CudfDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        roles = TextRole()
        new_df = self._subword_tokenize_columns(df)

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        df = dataset.data

        roles = TextRole()
        new_cols = [self._fname_prefix + "__" + col for col in df.columns]

        new_df = df.map_partitions(self._subword_tokenize_columns,
                                   meta=cudf.DataFrame(columns=new_cols).astype(str)).persist()

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform text dataset to subword_tokenized dataset (encode each token with number, bert-like input).
        E.g., "Nice weather today" -> "145 177 267"

        Args:
            dataset: Cudf or DaskCudf dataset of text features.

        Returns:
            Respective subword_tokenized dataset.

        """

        # checks here
        super().transform(dataset)
        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class ConcatTextTransformerGPU(LAMLTransformer):
    """Concat text features transformer (GPU).

    Args:
        special_token: Add special token between columns.
    """

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "concatedGPU"

    def __init__(self, special_token: str = " [SEP] "):
        self.special_token = special_token

    # concatenate text columns
    def _concat_columns(self, df):
        col_name = self._fname_prefix + "__" +\
                   "__".join(df.columns)
        outputs = df[df.columns[0]].fillna("").astype(str)
        for col in df.columns[1:]:
            outputs = outputs + f"{self.special_token}" +\
                      df[col].fillna("").astype(str)
        new_df = cudf.DataFrame({col_name: outputs})
        return new_df

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> CudfDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        roles = TextRole()
        new_df = self._concat_columns(df)
        # create resulted
        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in new_df.columns})
        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        df = dataset.data

        roles = TextRole()
        col_name = self._fname_prefix + "__" + "__".join(df.columns)
        new_df = df.map_partitions(self._concat_columns,
                                   meta=cudf.DataFrame(columns=[col_name]).astype(str)).persist()

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in new_df.columns})
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform text dataset to one text column.

        Args:
            dataset: Cudf or Daskcudf dataset of text features.

        Returns:
            Cudf or Daskcudf dataset with one text column.

        """

        # checks here
        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)
        

class AutoNLPWrapGPU(LAMLTransformer):
    """Calculate text embeddings (GPU).

    Args:
        model_name: Method for aggregating word embeddings into sentence embedding.
        transformer_params: Aggregating model parameters.
        embedding_model: 'fasttext' or 'bpe' - one of torch-nlp word embedding pre-trained models.
        word_vectors_cache: path to cached word embeddings for 'embedding_model'.
        cache_dir: If ``None`` - do not cache transformed datasets.
        bert_model: Name of HuggingFace transformer model.
        subs: Subsample to calculate freqs. If None - full data.
        multigpu: Use Data Parallel.
        random_state: Random state to take subsample.
        train_fasttext: Train fasttext. ### not used anymore
        fasttext_params: Fasttext init params. ### not used anymore
        fasttext_epochs: Number of epochs to train. ### not used anymore
        verbose: Verbosity.
        device: Torch device or str.
        **kwargs: Unused params.
    """

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "embGPU"
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
        word_vectors_cache: str = ".word_vectors_cache",
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
        lang: str = "en",
        **kwargs: Any,
    ):
        if train_fasttext:
            assert model_name in self._trainable, f"If train fasstext then model must be in {self._trainable}"

        assert model_name in self._names, f"Model name must be one of {self._names}"
        self.device = device
        self.multigpu = multigpu
        self.word_vectors_cache = word_vectors_cache if word_vectors_cache is not None else ".word_vectors_cache"
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.subs = subs
        self.train_fasttext = train_fasttext # outdated parameter, available word vectors are pretrained
        self.fasttext_epochs = fasttext_epochs
        if fasttext_params is not None:
            self.fasttext_params.update(fasttext_params)
        self.dicts = {}
        self.sent_scaler = sent_scaler
        self.verbose = verbose
        self.lang = lang
        self.model_name = model_name
        self.transformer_params = model_by_name[self.model_name]
        if transformer_params is not None:
            self.transformer_params.update(transformer_params)

        self._update_bert_model(bert_model)

        if embedding_model is not None:
            assert embedding_model in ["fasttext", "bpe"], f"embedding model must be in [fasttext, bpe]" \
                                                           f"but you passed {embedding_model}"
            if embedding_model == "fasttext":
                embedding_model = FastText(language=self.lang, cache=self.word_vectors_cache)
            elif embedding_model == "bpe":
                embedding_model = BPEmb(language=self.lang, dim=300, cache=self.word_vectors_cache)

            self.transformer_params = self._update_transformers_emb_model(self.transformer_params, embedding_model, 300)

        if self.model_name == "wat":
            self.transformer = WeightedAverageTransformerGPU
        else:
            self.transformer = DLTransformerGPU

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
        if self.model_name == "wat":
            params["embed_size"] = emb_size
            params["embedding_model"] = model
        elif self.model_name in {"random_lstm", "borep"}:
            params["dataset_params"]["embedding_model"] = model
            params["dataset_params"]["embed_size"] = emb_size
            params["model_params"]["embed_size"] = emb_size

        return params

    def _fit_cupy(self, dataset: GpuNumericalDataset):
        dataset = dataset.to_cudf()
        df = dataset.data

        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        names = []
        for n, i in enumerate(subs.columns):
            transformer_params = deepcopy(self.transformer_params)
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

    def _fit_daskcudf(self, dataset: GpuNumericalDataset):
        df = dataset.data

        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        names = []
        for n, i in enumerate(subs.columns):
            transformer_params = deepcopy(self.transformer_params)
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
                "transformer": deepcopy(transformer.fit(subs[i].compute())),
                "feats": feats,
            }
            names.extend(feats)
            logger.info3(f"Feature {i} fitted")

            del transformer
            gc.collect()
            torch.cuda.empty_cache()

        self._features = names
        return self

    def fit(self, dataset: GpuNumericalDataset):
        """Fit chosen transformer and create feature names.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of text features.

        """
        for check_func in self._fit_checks:
            check_func(dataset)

        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

        if isinstance(dataset, DaskCudfDataset):
            self._fit_daskcudf(dataset)
        else:
            self._fit_cupy(dataset)
        return self

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        roles = NumericRole()
        outputs = []
        for n, i in enumerate(df.columns):
            if self.cache_dir is not None:
                full_hash = get_textarr_hash(df[i].to_pandas()) + get_textarr_hash(self.dicts[i]["feats"])
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

            new_arr = cp.array(new_arr)
            output = dataset.empty().to_cupy()
            output.set_data(new_arr, self.dicts[i]["feats"], roles)
            outputs.append(output)
            logger.info3(f"Feature {i} transformed")
        # create resulted
        dataset = dataset.empty().to_cupy().concat(outputs)
        # instance-wise sentence embedding normalization
        dataset.data = dataset.data / self._sentence_norm(dataset.data, self.sent_scaler)
        return dataset

    def _transform_daskcudf(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        df = dataset.data
        gpu_cnt = torch.cuda.device_count()

        # transform
        roles = NumericRole()
        outputs = []
        for n, i in enumerate(df.columns):
            if self.cache_dir is not None:
                full_hash = get_textarr_hash(df[i].compute().to_pandas()) + get_textarr_hash(self.dicts[i]["feats"])
                fname = os.path.join(self.cache_dir, full_hash + ".pkl")
                if os.path.exists(fname):
                    logger.info3(f"Load saved dataset for {i}")
                    with open(fname, "rb") as f:
                        new_arr = pickle.load(f)
                else:
                    new_arr = self.dicts[i]["transformer"].transform(df[i].compute())
                    with open(fname, "wb") as f:
                        pickle.dump(new_arr, f)
            else:
                new_arr = self.dicts[i]["transformer"].transform(df[i].compute())

            new_df = cudf.DataFrame(new_arr, columns=[f"{i}_{k}" for k in range(new_arr.shape[1])])
            new_df = dask_cudf.from_cudf(new_df, npartitions=gpu_cnt).persist()

            output = dataset.empty()
            output.set_data(new_df, [f"{i}_{k}" for k in range(new_df.shape[1])], roles)
            outputs.append(output)
            logger.info3(f"Feature {i} transformed")
        # create resulted

        dataset = dataset.empty().concat(outputs)
        # instance-wise sentence embedding normalization
        dataset.data = dataset.data.map_partitions(self._sentence_norm_map, self.sent_scaler).persist()

        return dataset

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform tokenized dataset to text embeddings.

        Args:
            dataset: Cudf or Daskcudf dataset of text features.

        Returns:
            Cupy or Cudf or Daskcudf dataset with text embeddings.

        """

        # checks here
        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)

    @staticmethod
    def _sentence_norm(x: cp.ndarray, mode: Optional[str] = None) -> Union[cp.ndarray, float]:
        """Get sentence embedding norm."""
        if mode == "l2":
            return ((x ** 2).sum(axis=1, keepdims=True)) ** 0.5
        elif mode == "l1":
            return cp.abs(x).sum(axis=1, keepdims=True)
        if mode is not None:
            logger.info2("Unknown sentence scaler mode: sent_scaler={}, " "no normalization will be used".format(mode))
            pass
        return 1

    @staticmethod
    def _sentence_norm_map(df: cudf.DataFrame, mode: Optional[str] = None):
        """Get sentence embedding norm and normalize cudf.DataFrame"""
        x = df.values
        if mode == "l2":
            norm = ((x ** 2).sum(axis=1, keepdims=True)) ** 0.5
        elif mode == "l1":
            norm = cp.abs(x).sum(axis=1, keepdims=True)
        else:
            norm = 1
        return cudf.DataFrame(x / norm, index=df.index, columns=df.columns)
