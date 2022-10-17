"""Text features transformers."""

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


# try:
#     import gensim
# except:
#     import warnings
#
#     warnings.warn("'gensim' - package isn't installed")

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

# necessary for custom torch.svd_lowrank realization
# from torch import Tensor
# from torch import _linalg_utils as _utils

from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.feature_extraction.text import CountVectorizer
# from cuml.dask.feature_extraction.text import TfidfTransformer as TfidfTransformer_mgpu
# from cuml.dask.common import to_sparse_dask_array

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.gpu.gpu_dataset import CupySparseDataset
from lightautoml.dataset.roles import TextRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.text.gpu.tokenizer_gpu import BaseTokenizer_gpu
from lightautoml.text.gpu.tokenizer_gpu import SimpleEnTokenizer_gpu
from lightautoml.text.gpu.tokenizer_gpu import SimpleRuTokenizer_gpu
from lightautoml.text.gpu.dl_transformers_gpu import BOREP_gpu
from lightautoml.text.gpu.dl_transformers_gpu import BertEmbedder_gpu
from lightautoml.text.gpu.dl_transformers_gpu import DLTransformer_gpu
from lightautoml.text.gpu.dl_transformers_gpu import RandomLSTM_gpu
from lightautoml.text.gpu.embed_dataset_gpu import EmbedDataset_gpu
from lightautoml.text.gpu.embed_dataset_gpu import BertDataset_gpu
from lightautoml.text.utils import get_textarr_hash
from lightautoml.text.gpu.weighted_average_transformer_gpu import WeightedAverageTransformer_gpu

from lightautoml.transformers.gpu.svd_utils_gpu import _svd_lowrank

from lightautoml.transformers.text import oof_task_check
from lightautoml.transformers.text import text_check
from lightautoml.transformers.text import TunableTransformer

logger = logging.getLogger(__name__)

GpuNumericalDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]

model_by_name = {
    "random_lstm": {
        "model": RandomLSTM_gpu,
        "model_params": {
            "embed_size": 300,
            "hidden_size": 256,
            "pooling": "mean",
            "num_layers": 1,
        },
        "dataset": EmbedDataset_gpu,
        "dataset_params": {
            "embedding_model": None,
            "max_length": 200,
            "embed_size": 300,
        },
        "loader_params": {"batch_size": 1024, "shuffle": False, "num_workers": 4},
        "embedding_model_params": {},
    },
    "random_lstm_bert": {
        "model": RandomLSTM_gpu,
        "model_params": {
            "embed_size": 768,
            "hidden_size": 256,
            "pooling": "mean",
            "num_layers": 1,
        },
        "dataset": BertDataset_gpu,
        "dataset_params": {"max_length": 256, "model_name": "bert-base-cased"},
        "loader_params": {"batch_size": 320, "shuffle": False, "num_workers": 4},
        "embedding_model": BertEmbedder_gpu,
        "embedding_model_params": {"model_name": "bert-base-cased", "pooling": "none"},
    },
    "borep": {
        "model": BOREP_gpu,
        "model_params": {
            "embed_size": 300,
            "proj_size": 300,
            "pooling": "mean",
            "max_length": 200,
            "init": "orthogonal",
            "pos_encoding": False,
        },
        "dataset": EmbedDataset_gpu,
        "dataset_params": {
            "embedding_model": None,
            "max_length": 200,
            "embed_size": 300,
        },
        "loader_params": {"batch_size": 1024, "shuffle": False, "num_workers": 4},
        "embedding_model_params": {},
    },
    "pooled_bert": {
        "model": BertEmbedder_gpu,
        "model_params": {"model_name": "bert-base-cased", "pooling": "mean"},
        "dataset": BertDataset_gpu,
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


class TunableTransformer_gpu(LAMLTransformer):
    """Base class for ML transformers (GPU).

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

    def init_params_on_input(self, dataset: GpuNumericalDataset) -> dict:
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


class TfidfTextTransformer_gpu(TunableTransformer_gpu):
    """Simple Tfidf vectorizer (GPU)."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tfidf_gpu"
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
        """

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
        client: Any = None
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

    # convert csr sparse matrix to coo format
    def _csr2coo(self, arr):
        coo = arr.tocoo()
        i = torch.as_tensor(cp.vstack((coo.row, coo.col)), dtype=torch.int64, device='cuda')
        v = torch.as_tensor(coo.data, dtype=torch.float32, device='cuda')
        coo = torch.cuda.sparse.FloatTensor(i, v, torch.Size(coo.shape))
        return coo

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
                                   meta=cupyx.scipy.sparse.csr.csr_matrix(
                                       (len(df.partitions[0]), self._default_params["max_features"]))).persist()

        rs = da.random.RandomState(RandomState=cp.random.RandomState)
        _, _, Vh = da.linalg.svd_compressed(output, k=self.n_components, seed=rs,
                                            n_oversamples=self.n_oversample)

        self.Vh = cupyx.scipy.sparse.csr_matrix(Vh.T.compute())
        return self

    def fit(self, dataset: GpuNumericalDataset):
        """Fit tfidf vectorizer.

        Args:
            dataset: Pandas or Numpy dataset of text features.

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
        """Transform text dataset to sparse tfidf representation.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of text features.

        Returns:
            Sparse dataset with encoded text.

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
                                   meta=cupyx.scipy.sparse.csr.csr_matrix(
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
        """Transform text dataset to sparse tfidf representation.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of text features.

        Returns:
            Sparse dataset with encoded text.

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


class TokenizerTransformer_gpu(LAMLTransformer):
    """Simple tokenizer transformer (GPU)."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tokenized_gpu"

    def __init__(self, tokenizer: BaseTokenizer_gpu = SimpleEnTokenizer_gpu()):
        """

        Args:
            tokenizer: text tokenizer.

        """
        self.tokenizer = tokenizer

    def tokenize_columns(self, df):
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
        new_df = self.tokenize_columns(df)

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        df = dataset.data
        
        roles = TextRole()
        new_cols = [self._fname_prefix + "__" + col for col in df.columns]

        # Since stemmer works on CPU, here is an unpretty workaround
        if self.tokenizer.stemmer is None:
            new_df = df.map_partitions(self.tokenize_columns, meta=cudf.DataFrame(columns=new_cols).astype(str)).persist()
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
            dataset: Cudf or Cupy or DaskCudf dataset of text features.

        Returns:
            Respective dataset with tokenized text.

        """

        # checks here
        super().transform(dataset)
        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class SubwordTokenizerTransformer_gpu(LAMLTransformer):
    """Subword tokenizer transformer (GPU)."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tokenized_gpu"

    def __init__(self, vocab_path: str = None, data_path: Any = None, is_hash: bool = False, max_length: int = 300,
                 tokenizer: str = "bpe", vocab_size: int = 30000,
                 save_path: str = None):
        """

        Args:
            vocab_path: path to vocabulary .txt file,
            data_path: .txt file (saved pd.Series) for the tokenizer to be trained on (if vocab is not specified)
            is_hash: True means vocab is not raw vocab but was transformed with hash_vocab function from cudf,
            max_length: max number of tokens to leave in one text (exceeding ones would be truncated)
            tokenizer: ["bpe" or "wordpiece"] if vocab is None. Type of tokenizer to be trained
            vocab_size: vocabulary size for trained tokenizer
            save_path: path where trained vocabulary would be saved to

        """
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

            with open(save_path, 'w+') as f:
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

    @staticmethod
    def concat_columns(df, col_name):
        outputs = df[df.columns[0]].fillna("").astype(str)
        for col in df.columns[1:]:
            outputs = outputs + " " + df[col].fillna("").astype(str)
        new_df = cudf.DataFrame({col_name: outputs})
        return new_df

    def subword_tokenize_columns(self, df):
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
            pred = self.concat_columns(cudf.DataFrame(pred['input_ids']).astype(str), col_name)
            pred = pred[col_name].str.replace_tokens(self.targets, "").str.normalize_spaces()
            outputs.append(pred)
        new_df = cudf.DataFrame(cudf.concat(outputs, axis=1))
        return new_df

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> CudfDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        roles = TextRole()
        new_df = self.subword_tokenize_columns(df)

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        df = dataset.data

        roles = TextRole()
        new_cols = [self._fname_prefix + "__" + col for col in df.columns]

        new_df = df.map_partitions(self.subword_tokenize_columns,
                                   meta=cudf.DataFrame(columns=new_cols).astype(str)).persist()

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in self.features})
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform text dataset to tokenized text dataset.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of text features.

        Returns:
            Respective dataset with tokenized text.

        """

        # checks here
        super().transform(dataset)
        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class ConcatTextTransformer_gpu(LAMLTransformer):
    """Concat text features transformer (GPU)."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "concated_gpu"

    def __init__(self, special_token: str = " [SEP] "):
        """

        Args:
            special_token: Add special token between columns.

        """
        self.special_token = special_token

    def concat_columns(self, df):
        col_name = self._fname_prefix + "__" + "__".join(df.columns)
        outputs = df[df.columns[0]].fillna("").astype(str)
        for col in df.columns[1:]:
            outputs = outputs + f"{self.special_token}" + df[col].fillna("").astype(str)
        new_df = cudf.DataFrame({col_name: outputs})
        return new_df

    def _transform_cupy(self, dataset: GpuNumericalDataset) -> CudfDataset:
        dataset = dataset.to_cudf()
        df = dataset.data

        roles = TextRole()
        new_df = self.concat_columns(df)
        # create resulted
        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in new_df.columns})
        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        df = dataset.data

        roles = TextRole()
        col_name = self._fname_prefix + "__" + "__".join(df.columns)
        new_df = df.map_partitions(self.concat_columns,
                                   meta=cudf.DataFrame(columns=[col_name]).astype(str)).persist()

        output = dataset.empty()
        output.set_data(new_df, None, {feat: roles for feat in new_df.columns})
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform text dataset to one text column.

        Args:
            dataset: Cupy or Cudf or Daskcudf dataset of text features.

        Returns:
            Pandas dataset with one text column.

        """

        # checks here
        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)
        

class AutoNLPWrap_gpu(LAMLTransformer):
    """Calculate text embeddings (GPU)."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "emb_gpu"
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
        lang: str = "en",
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
                embedding_model = FastText(language=self.lang)
            elif embedding_model == "bpe":
                embedding_model = BPEmb(language=self.lang, dim=300)

            self.transformer_params = self._update_transformers_emb_model(self.transformer_params, embedding_model, 300)

        # elif type(embedding_model) == str:
        #     self.transformer_params = self._update_transformers_emb_model(self.transformer_params, embedding_model, 300)
        # if embedding_model is not 'sent_scaler': 'l2'None:
        #     if isinstance(embedding_model, str):
        #         try:
        #             embedding_model = gensim.models.FastText.load(embedding_model)
        #         except:
        #             try:
        #                 embedding_model = gensim.models.FastText.load_fasttext_format(embedding_model)
        #             except:
        #                 embedding_model = gensim.models.KeyedVectors.load(embedding_model)
        #
        #     self.transformer_params = self._update_transformers_emb_model(self.transformer_params, embedding_model)
        #
        # else:
        #
        #     self.train_fasttext = self.model_name in self._trainable

        if self.model_name == "wat":
            self.transformer = WeightedAverageTransformer_gpu
        else:
            self.transformer = DLTransformer_gpu

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
        # if emb_size is None:
        #     try:
        #         # Gensim checker [1]
        #         emb_size = model.vector_size
        #     except:
        #         try:
        #             # Gensim checker[2]
        #             emb_size = model.vw.vector_size
        #         except:
        #             try:
        #                 # Natasha checker
        #                 emb_size = model[model.vocab.words[0]].shape[0]
        #             except:
        #                 try:
        #                     # Dict of embeddings checker
        #                     emb_size = next(iter(model.values())).shape[0]
        #                 except:
        #                     raise ValueError("Unrecognized embedding dimention, please specify it in model_params")
        # try:
        #     model = model.wv
        # except:
        #     pass

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
            # if self.train_fasttext:
            #     embedding_model = gensim.models.FastText(**self.fasttext_params)
            #     common_texts = [i.split(" ") for i in subs[i].values]
            #     # common_texts = []
            #     # for k in range(subs[i].shape[0]):
            #     #     common_texts.append(subs[i][k].split(" "))
            #     # common_texts = [i.split(" ") for i in subs[i].to_pandas().values]
            #     embedding_model.build_vocab(corpus_iterable=common_texts)
            #     embedding_model.train(
            #         corpus_iterable=common_texts,
            #         total_examples=len(common_texts),
            #         epochs=self.fasttext_epochs,
            #     )
            #     transformer_params = self._update_transformers_emb_model(transformer_params, embedding_model)
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
#             logger.info3(f"Feature {i} fitted")

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
            # if self.train_fasttext:
            #     embedding_model = gensim.models.FastText(**self.fasttext_params)
            #     common_texts = []
            #     for k in range(subs[i].shape[0].compute()):
            #         common_texts.append(subs[i][k].compute().iloc[0].split(" "))
            #     embedding_model.build_vocab(corpus_iterable=common_texts)
            #     embedding_model.train(
            #         corpus_iterable=common_texts,
            #         total_examples=len(common_texts),
            #         epochs=self.fasttext_epochs,
            #     )
            #     transformer_params = self._update_transformers_emb_model(transformer_params, embedding_model)

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
#             logger.info3(f"Feature {i} fitted")

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
#                     logger.info3(f"Load saved dataset for {i}")
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
#             logger.info3(f"Feature {i} transformed")
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
#                     logger.info3(f"Load saved dataset for {i}")
                    with open(fname, "rb") as f:
                        new_arr = pickle.load(f)
                else:
                    new_arr = self.dicts[i]["transformer"].transform(df[i].compute())
                    with open(fname, "wb") as f:
                        pickle.dump(new_arr, f)
            else:
                new_arr = self.dicts[i]["transformer"].transform(df[i].compute())

            new_df = cudf.DataFrame(new_arr)
            new_df = dask_cudf.from_cudf(new_df, npartitions=gpu_cnt)

            output = dataset.empty()
            # output.set_data(new_df, self.dicts[i]["feats"], roles)
            output.set_data(new_df, None, roles)
            outputs.append(output)
#             logger.info3(f"Feature {i} transformed")
        # create resulted

        dataset = dataset.empty().concat(outputs)
        # instance-wise sentence embedding normalization
        dataset.data = dataset.data.map_partitions(self._sentence_norm_map, self.sent_scaler).persist()

        return dataset

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform tokenized dataset to text embeddings.

        Args:
            dataset: Cupy or Cudf or Daskcudf dataset of text features.

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
            pass
#             logger.info2("Unknown sentence scaler mode: sent_scaler={}, " "no normalization will be used".format(mode))
        return 1

    @staticmethod
    def _sentence_norm_map(df: cudf.DataFrame, mode: Optional[str] = None):
        x = df.values
        if mode == "l2":
            norm = ((x ** 2).sum(axis=1, keepdims=True)) ** 0.5
        elif mode == "l1":
            norm = cp.abs(x).sum(axis=1, keepdims=True)
        else:
            norm = 1
        return cudf.DataFrame(x / norm, index=df.index, columns=df.columns)
