"""Categorical features transformerrs."""

from itertools import combinations
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from typing import cast

import numpy as np

from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.murmurhash import murmurhash3_32

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import CSRSparseDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import CategoryRole
from ..dataset.roles import NumericRole
from .base import LAMLTransformer


# type - something that can be convered to pandas dataset
NumpyOrPandas = Union[NumpyDataset, PandasDataset]
NumpyOrSparse = Union[NumpyDataset, CSRSparseDataset]


def categorical_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Raises AssertionError if non-categorical features are present.

    Args:
        dataset: LAMLDataset to check.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[f].name == "Category", "Only categories accepted in this transformer"


def oof_task_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Input.

    """
    task = dataset.task
    assert task.name in [
        "binary",
        "reg",
    ], "Only binary and regression tasks supported in this transformer"


def multiclass_task_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Input.

    Returns:

    """
    task = dataset.task
    assert task.name in ["multiclass"], "Only multiclass tasks supported in this transformer"


def encoding_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Input.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[
            f
        ].label_encoded, "Transformer should be applied to category only after label encoding. Feat {0} is {1}".format(
            f, roles[f]
        )


class LabelEncoder(LAMLTransformer):
    """Simple LabelEncoder in order of frequency.

    Labels are integers from 1 to n. Unknown category encoded as 0.
    NaN is handled as a category value.

    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "le"

    # _output_role = CategoryRole(np.int32, label_encoded=True)
    _fillna_val = 0

    def __init__(self, subs: Optional[int] = None, random_state: int = 42):
        """

        Args:
            subs: Subsample to calculate freqs. If None - full data.
            random_state: Random state to take subsample.

        """
        self.subs = subs
        self.random_state = random_state
        self._output_role = CategoryRole(np.int32, label_encoded=True)

    def _get_df(self, dataset: NumpyOrPandas) -> DataFrame:
        """Get df and sample.

        Args:
            dataset: Input dataset.

        Returns:
            Subsample.

        """
        dataset = dataset.to_pandas()
        df = dataset.data

        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        return subs

    def fit(self, dataset: NumpyOrPandas):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        roles = dataset.roles
        subs = self._get_df(dataset)

        self.dicts = {}
        for i in subs.columns:
            role = roles[i]
            # TODO: think what to do with this warning
            co = role.unknown
            cnts = (
                subs[i]
                .value_counts(dropna=False)
                .reset_index()
                .sort_values([i, "index"], ascending=[False, True])
                .set_index("index")
            )
            vals = cnts[cnts > co].index.values
            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals)

        return self

    def transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Transform categorical dataset to int labels.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        new_arr = np.empty(dataset.shape, dtype=self._output_role.dtype)

        for n, i in enumerate(df.columns):
            # to be compatible with OrdinalEncoder
            if i in self.dicts:
                new_arr[:, n] = df[i].map(self.dicts[i]).fillna(self._fillna_val).values
            else:
                new_arr[:, n] = df[i].values.astype(self._output_role.dtype)

        # create resulted
        output = dataset.empty().to_numpy()
        output.set_data(new_arr, self.features, self._output_role)

        return output


class OHEEncoder(LAMLTransformer):
    """
    Simple OneHotEncoder over label encoded categories.
    """

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        make_sparse: Optional[bool] = None,
        total_feats_cnt: Optional[int] = None,
        dtype: type = np.float32,
    ):
        """

        Args:
            make_sparse: Create sparse matrix.
            total_feats_cnt: Initial features number.
            dtype: Dtype of new features.

        """
        self.make_sparse = make_sparse
        self.total_feats_cnt = total_feats_cnt
        self.dtype = dtype

        if self.make_sparse is None:
            assert self.total_feats_cnt is not None, "Param total_feats_cnt should be defined if make_sparse is None"

    def fit(self, dataset: NumpyOrPandas):
        """Calc output shapes.

        Automatically do ohe in sparse form if approximate fill_rate < `0.2`.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        max_idx = data.max(axis=0)
        min_idx = data.min(axis=0)

        # infer make sparse
        if self.make_sparse is None:
            fill_rate = self.total_feats_cnt / (self.total_feats_cnt - max_idx.shape[0] + max_idx.sum())
            self.make_sparse = fill_rate < 0.2

        # create ohe
        self.ohe = OneHotEncoder(
            categories=[np.arange(x, y + 1, dtype=np.int32) for (x, y) in zip(min_idx, max_idx)],
            # drop=np.ones(max_idx.shape[0], dtype=np.int32),
            dtype=self.dtype,
            sparse=self.make_sparse,
            handle_unknown="ignore",
        )
        self.ohe.fit(data)

        features = []
        for cats, name in zip(self.ohe.categories_, dataset.features):
            # cats = cats[cats != 1]
            features.extend(["ohe_{0}__{1}".format(x, name) for x in cats])

        self._features = features

        return self

    def transform(self, dataset: NumpyOrPandas) -> NumpyOrSparse:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        # transform
        data = self.ohe.transform(data)

        # create resulted
        output = dataset.empty()
        if self.make_sparse:
            output = output.to_csr()

        output.set_data(data, self.features, NumericRole(self.dtype))
        return output


class FreqEncoder(LabelEncoder):
    """
    Labels are encoded with frequency in train data.

    Labels are integers from 1 to n. Unknown category encoded as 1.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"

    # _output_role = NumericRole(np.float32)
    _fillna_val = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(np.float32)

    def fit(self, dataset: NumpyOrPandas):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Pandas or Numpy dataset of categorical features

        Returns:
            self.

        """
        # set transformer names and add checks
        LAMLTransformer.fit(self, dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        self.dicts = {}
        for i in df.columns:
            # we make assertion in checks, so cast is ok
            # TODO: think what to do with this warning
            cnts = df[i].value_counts(dropna=False)
            self.dicts[i] = cnts[cnts > 1]

        return self


class TargetEncoder(LAMLTransformer):
    """
    Out-of-fold target encoding.

    Limitation:

        - Required .folds attribute in dataset - array of int from 0 to n_folds-1.
        - Working only after label encoding.

    """

    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(self, alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0)):
        """

        Args:
            alphas: Smooth coefficients.

        """
        self.alphas = alphas

    @staticmethod
    def binary_score_func(candidates: np.ndarray, target: np.ndarray) -> int:
        """Score candidates alpha with logloss metric.

        Args:
            candidates: Candidate oof encoders.
            target: Target array.

        Returns:
            Index of best encoder.

        """
        target = target[:, np.newaxis]
        scores = -(target * np.log(candidates) + (1 - target) * np.log(1 - candidates)).mean(axis=0)
        idx = scores.argmin()

        return idx

    @staticmethod
    def reg_score_func(candidates: np.ndarray, target: np.ndarray) -> int:
        """Score candidates alpha with mse metric.

        Args:
            candidates: Candidate oof encoders.
            target: Target array.

        Returns:
            Index of best encoder.

        """
        target = target[:, np.newaxis]
        scores = ((target - candidates) ** 2).mean(axis=0)
        idx = scores.argmin()

        return idx

    def fit(self, dataset: NumpyOrPandas):
        super().fit_transform(dataset)

    def fit_transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Calc oof encoding and save encoding stats for new data.

        Args:
            dataset: Pandas or Numpy dataset of
              categorical label encoded features.

        Returns:
            NumpyDataset - target encoded features.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        target = dataset.target.astype(np.int32)
        score_func = self.binary_score_func if dataset.task.name == "binary" else self.reg_score_func

        folds = dataset.folds
        n_folds = folds.max() + 1
        alphas = np.array(self.alphas)[np.newaxis, :]

        self.encodings = []
        prior = target.mean()
        # folds priors
        f_sum = np.zeros(n_folds, dtype=np.float64)
        f_count = np.zeros(n_folds, dtype=np.float64)

        np.add.at(f_sum, folds, target)
        np.add.at(f_count, folds, 1)

        folds_prior = (f_sum.sum() - f_sum) / (f_count.sum() - f_count)
        oof_feats = np.zeros(data.shape, dtype=np.float32)

        for n in range(data.shape[1]):
            vec = data[:, n]

            # calc folds stats
            enc_dim = vec.max() + 1
            f_sum = np.zeros((enc_dim, n_folds), dtype=np.float64)
            f_count = np.zeros((enc_dim, n_folds), dtype=np.float64)

            np.add.at(f_sum, (vec, folds), target)
            np.add.at(f_count, (vec, folds), 1)

            # calc total stats
            t_sum = f_sum.sum(axis=1, keepdims=True)
            t_count = f_count.sum(axis=1, keepdims=True)

            # calc oof stats
            oof_sum = t_sum - f_sum
            oof_count = t_count - f_count
            # calc candidates alpha
            candidates = (
                (oof_sum[vec, folds, np.newaxis] + alphas * folds_prior[folds, np.newaxis])
                / (oof_count[vec, folds, np.newaxis] + alphas)
            ).astype(np.float32)
            idx = score_func(candidates, target)

            # write best alpha
            oof_feats[:, n] = candidates[:, idx]
            # calc best encoding
            enc = ((t_sum[:, 0] + alphas[0, idx] * prior) / (t_count[:, 0] + alphas[0, idx])).astype(np.float32)

            self.encodings.append(enc)

        output = dataset.empty()
        self.output_role = NumericRole(np.float32, prob=output.task.name == "binary")
        output.set_data(oof_feats, self.features, self.output_role)

        return output

    def transform(self, dataset: NumpyOrPandas) -> NumpyOrSparse:
        """Transform categorical dataset to target encoding.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        # transform
        out = np.zeros(data.shape, dtype=np.float32)
        for n, enc in enumerate(self.encodings):
            out[:, n] = enc[data[:, n]]

        # create resulted
        output = dataset.empty()
        output.set_data(out, self.features, self.output_role)

        return output


class MultiClassTargetEncoder(LAMLTransformer):
    """
    Out-of-fold target encoding for multiclass task.

    Limitation:

        - Required .folds attribute in dataset - array of int from 0 to n_folds-1.
        - Working only after label encoding

    """

    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self, alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0)):
        self.alphas = alphas

    @staticmethod
    def score_func(candidates: np.ndarray, target: np.ndarray) -> int:
        """


        Args:
            candidates: np.ndarray.
            target: np.ndarray.

        Returns:
            index of best encoder.

        """
        target = target[:, np.newaxis, np.newaxis]
        scores = -np.log(np.take_along_axis(candidates, target, axis=1)).mean(axis=0)[0]
        idx = scores.argmin()

        return idx

    def fit_transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Pandas or Numpy dataset of categorical label encoded features.

        Returns:
            NumpyDataset - target encoded features.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data
        target = dataset.target.astype(np.int32)
        n_classes = target.max() + 1
        self.n_classes = n_classes

        folds = dataset.folds
        n_folds = folds.max() + 1
        alphas = np.array(self.alphas)[np.newaxis, np.newaxis, :]

        self.encodings = []
        # prior
        prior = cast(np.ndarray, np.arange(n_classes)[:, np.newaxis] == target).mean(axis=1)
        # folds prior
        f_sum = np.zeros((n_classes, n_folds), dtype=np.float64)
        f_count = np.zeros((1, n_folds), dtype=np.float64)

        np.add.at(f_sum, (target, folds), 1)
        np.add.at(f_count, (0, folds), 1)

        # N_classes x N_folds
        folds_prior = ((f_sum.sum(axis=1, keepdims=True) - f_sum) / (f_count.sum(axis=1, keepdims=True) - f_count)).T
        oof_feats = np.zeros(data.shape + (n_classes,), dtype=np.float32)

        self._features = []
        for i in dataset.features:
            for j in range(n_classes):
                self._features.append("{0}_{1}__{2}".format("multioof", j, i))

        for n in range(data.shape[1]):
            vec = data[:, n]

            # calc folds stats
            enc_dim = vec.max() + 1
            f_sum = np.zeros((enc_dim, n_classes, n_folds), dtype=np.float64)
            f_count = np.zeros((enc_dim, 1, n_folds), dtype=np.float64)

            np.add.at(f_sum, (vec, target, folds), 1)
            np.add.at(f_count, (vec, 0, folds), 1)

            # calc total stats
            t_sum = f_sum.sum(axis=2, keepdims=True)
            t_count = f_count.sum(axis=2, keepdims=True)

            # calc oof stats
            oof_sum = t_sum - f_sum
            oof_count = t_count - f_count

            # (N x N_classes x 1 + 1 x 1 x N_alphas * N x N_classes x 1) / (N x 1 x 1 + N x 1 x 1) -> N x N_classes x N_alphas
            candidates = (
                (oof_sum[vec, :, folds, np.newaxis] + alphas * folds_prior[folds, :, np.newaxis])
                / (oof_count[vec, :, folds, np.newaxis] + alphas)
            ).astype(np.float32)

            # norm over 1 axis
            candidates /= candidates.sum(axis=1, keepdims=True)

            idx = self.score_func(candidates, target)
            oof_feats[:, n] = candidates[..., idx]
            enc = ((t_sum[..., 0] + alphas[0, 0, idx] * prior) / (t_count[..., 0] + alphas[0, 0, idx])).astype(
                np.float32
            )
            enc /= enc.sum(axis=1, keepdims=True)

            self.encodings.append(enc)

        output = dataset.empty()
        output.set_data(
            oof_feats.reshape((data.shape[0], -1)),
            self.features,
            NumericRole(np.float32, prob=True),
        )

        return output

    def transform(self, dataset: NumpyOrPandas) -> NumpyOrSparse:
        """Transform categorical dataset to target encoding.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_numpy()
        data = dataset.data

        # transform
        out = np.zeros(data.shape + (self.n_classes,), dtype=np.float32)
        for n, enc in enumerate(self.encodings):
            out[:, n] = enc[data[:, n]]

        out = out.reshape((data.shape[0], -1))

        # create resulted
        output = dataset.empty()
        output.set_data(out, self.features, NumericRole(np.float32, prob=True))

        return output


class CatIntersectstions(LabelEncoder):
    """Build label encoded intertsections of categorical variables."""

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "inter"

    def __init__(
        self,
        subs: Optional[int] = None,
        random_state: int = 42,
        intersections: Optional[Sequence[Sequence[str]]] = None,
        max_depth: int = 2,
    ):
        """Create label encoded intersection columns for categories.

        Args:
            intersections: Columns to create intersections.
              Default is None - all.
            max_depth: Max intersection depth.

        """
        super().__init__(subs, random_state)
        self.intersections = intersections
        self.max_depth = max_depth

    @staticmethod
    def _make_category(df: DataFrame, cols: Sequence[str]) -> np.ndarray:
        """Make hash for category interactions.

        Args:
            df: Input DataFrame
            cols: List of columns

        Returns:
            Hash np.ndarray.

        """
        res = np.empty((df.shape[0],), dtype=np.int32)

        for n, inter in enumerate(zip(*(df[x] for x in cols))):
            h = murmurhash3_32("_".join(map(str, inter)), seed=42)
            res[n] = h

        return res

    def _build_df(self, dataset: NumpyOrPandas) -> PandasDataset:
        """

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Dataset.
        """
        dataset = dataset.to_pandas()
        df = dataset.data

        roles = {}
        new_df = DataFrame(index=df.index)
        for comb in self.intersections:
            name = "({0})".format("__".join(comb))
            new_df[name] = self._make_category(df, comb)
            roles[name] = CategoryRole(
                object,
                unknown=max((dataset.roles[x].unknown for x in comb)),
                label_encoded=True,
            )

        output = dataset.empty()
        output.set_data(new_df, new_df.columns, roles)

        return output

    def fit(self, dataset: NumpyOrPandas):
        """Create label encoded intersections and save mapping.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)

        if self.intersections is None:
            self.intersections = []
            for i in range(2, min(self.max_depth, len(dataset.features)) + 1):
                self.intersections.extend(list(combinations(dataset.features, i)))

        inter_dataset = self._build_df(dataset)
        return super().fit(inter_dataset)

    def transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Create label encoded intersections and apply mapping

        Args:
            dataset: Pandas or Numpy dataset of categorical features

        Returns:

        """
        inter_dataset = self._build_df(dataset)
        return super().transform(inter_dataset)


class OrdinalEncoder(LabelEncoder):
    """
    Encoding ordinal categories into numbers.
    Number type categories passed as is,
    object type sorted in ascending lexicographical order.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "ord"

    # _output_role = NumericRole(np.float32)
    _fillna_val = np.nan

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(np.float32)

    def fit(self, dataset: NumpyOrPandas):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        """
        # set transformer names and add checks
        LAMLTransformer.fit(self, dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        roles = dataset.roles
        subs = self._get_df(dataset)

        self.dicts = {}
        for i in subs.columns:
            role = roles[i]
            try:
                flg_number = np.issubdtype(role.dtype, np.number)
            except TypeError:
                flg_number = False

            if not flg_number:
                co = role.unknown
                cnts = subs[i].value_counts(dropna=True)
                cnts = cnts[cnts > co].reset_index()
                cnts = Series(cnts["index"].astype(str).rank().values, index=cnts["index"].values)
                cnts = cnts.append(Series([cnts.shape[0] + 1], index=[np.nan]))
                self.dicts[i] = cnts

        return self
