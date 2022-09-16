"""Roles guess on gpu."""

from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset, CupyDataset
from lightautoml.dataset.roles import CategoryRole, ColumnRole, NumericRole
from lightautoml.reader.utils import set_sklearn_folds
from lightautoml.transformers.base import (
    ChangeRoles,
    LAMLTransformer,
    SequentialTransformer,
)
from lightautoml.transformers.gpu.categorical_gpu import (
    FreqEncoder_gpu,
    LabelEncoder_gpu,
    MultiClassTargetEncoder_gpu,
    OrdinalEncoder_gpu,
    TargetEncoder_gpu,
)
from lightautoml.transformers.gpu.numeric_gpu import QuantileBinning_gpu

RolesDict = Dict[str, ColumnRole]
Encoder_gpu = Union[TargetEncoder_gpu, MultiClassTargetEncoder_gpu]
GpuFrame = Union[cudf.DataFrame]
GpuDataset = Union[CudfDataset, CupyDataset]


def ginic_gpu(actual: GpuFrame, pred: GpuFrame) -> float:
    """Denormalized gini calculation.

    Args:
        actual_pred: array with true and predicted values
        inds: list of indices for true and predicted values

    Returns:
        Metric value

    """
    n = actual.shape[0]
    a_s = actual[cp.argsort(pred)]
    a_c = a_s.cumsum()
    gini_sum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return gini_sum / n


def ginic_gpu_new(actual: GpuFrame, pred: GpuFrame, empty_slice) -> float:
    """Denormalized gini calculation.

    Args:
        actual_pred: array with true and predicted values
        inds: list of indices for true and predicted values

    Returns:
        Metric value

    """

    n = cp.sum(~empty_slice, axis=0)

    ids = cp.argsort(pred, axis=0)
    a_s = cp.take_along_axis(actual, ids, axis=0)

    a_c = a_s.cumsum(axis=0)

    a_c[cp.take_along_axis(empty_slice, ids, axis=0)] = 0
    gini_sum = a_c.sum(axis=0) / a_s.sum(axis=0) - (n + 1) / 2.0
    return gini_sum / n


def gini_normalizedc_gpu(a: GpuFrame, p: GpuFrame) -> float:
    """Calculated normalized gini.

    Args:
        a_p: array with true and predicted values

    Returns:
        Metric value.

    """
    out = ginic_gpu(a, p) / ginic_gpu(a, a)

    assert not cp.isnan(out), "gini index is givin nan, is that ok? {0} and {1}".format(
        a, p
    )
    return out


def gini_normalizedc_gpu_new(a: GpuFrame, p: GpuFrame, empty_slice) -> float:
    """Calculated normalized gini.

    Args:
        a_p: array with true and predicted values

    Returns:
        Metric value.

    """

    out = ginic_gpu_new(a, p, empty_slice) / ginic_gpu_new(a, a, empty_slice)

    # assert not cp.isnan(out), 'gini index is givin nan, is that ok? {0} and {1}'.format(a, p)
    return out


def gini_normalized_gpu(
    y: GpuFrame, target: GpuFrame, empty_slice: GpuFrame = None
) -> float:
    """Calculate normalized gini index for dataframe data.

    Args:
        y: data.
        true_cols: columns with true data.
        pred_cols: columns with predict data.
        empty_slice: Mask.

    Returns:
        Gini value.

    """

    if empty_slice is None:
        empty_slice = cp.isnan(y)
    all_true = empty_slice.all()
    if all_true:
        return 0.0

    sl = ~empty_slice

    outp_size = 1 if target.ndim <= 1 else target.shape[1]
    pred_size = 1 if y.ndim <= 1 else y.shape[1]

    ginis = cp.zeros((outp_size,), dtype=cp.float32)

    for i in range(outp_size):
        j = min(i, pred_size - 1)
        yp = None
        if pred_size == 1:
            yp = y[sl]
        else:
            yp = y[:, j][sl]
        yt = None
        if outp_size == 1:
            yt = target[sl]
        else:
            yt = target[:, i][sl]

        ginis[i] = gini_normalizedc_gpu(yt, yp)

    return cp.abs(ginis).mean()


def gini_normalized_gpu_new(
    y: GpuFrame, target: GpuFrame, empty_slice: GpuFrame = None
) -> float:
    """Calculate normalized gini index for dataframe data.

    Args:
        y: data.
        true_cols: columns with true data.
        pred_cols: columns with predict data.
        empty_slice: Mask.

    Returns:
        Gini value.

    """

    if empty_slice is None:
        empty_slice = cp.isnan(y)
    all_true = empty_slice.all()
    if all_true:
        return 0.0

    sl = empty_slice
    sl = sl.reshape(sl.shape[0], -1)

    outp_size = 1 if target.ndim <= 1 else target.shape[1]
    pred_size = 1 if y.ndim <= 1 else y.shape[1]

    index_i = cp.arange(pred_size, dtype=cp.int32)
    index_i = cp.repeat(index_i, outp_size)

    index_j = cp.arange(len(index_i))

    yp_new = y.reshape(y.shape[0], -1)
    yp_new[sl] = 0

    yt_new = cp.repeat(target.reshape(target.shape[0], -1), pred_size, axis=0).reshape(
        target.shape[0], -1
    )
    yt_new[cp.repeat(sl, outp_size, axis=1)] = 0

    row_col_const = 20000000
    batch_size = row_col_const // yt_new.shape[0]

    ginis_new = []

    for i in range((index_j.shape[0] // batch_size) + 1):
        end = min((i + 1) * batch_size, index_j.shape[0])
        ginis_new.append(
            gini_normalizedc_gpu_new(
                yt_new[:, index_j[i * batch_size : end]],
                yp_new[:, index_i[i * batch_size : end]],
                sl[:, index_i[i * batch_size : end]],
            )
        )
    ginis_new = cp.concatenate(ginis_new)
    ginis_new = ginis_new.reshape((pred_size, outp_size)).astype(cp.float32)

    return cp.abs(ginis_new).mean(axis=1)


def get_target_and_encoder_gpu(train: GpuDataset) -> Tuple[Any, type]:
    """Get target encoder and target based on dataset.

    Args:
        train: Dataset.

    Returns:
        (Target values, Target encoder).

    """

    target = train.target
    if isinstance(target, cudf.Series):
        target = target.values

    target_name = train.target.name
    if train.task.name == "multiclass":
        n_out = cp.max(target) + 1
        target = target[:, cp.newaxis] == cp.arange(n_out)[cp.newaxis, :]
        encoder = MultiClassTargetEncoder_gpu
    else:
        encoder = TargetEncoder_gpu
    return target, encoder


def calc_ginis_gpu(
    data: Union[GpuFrame, cp.ndarray],
    target: Union[GpuFrame, cp.ndarray],
    empty_slice: Union[GpuFrame, cp.ndarray] = None,
) -> cp.ndarray:
    """

    Args:
        data: cp.ndarray or gpu DataFrame.
        target: cp.ndarray or gpu DataFrame.
        empty_slice: cp.ndarray or gpu DataFrame.

    Returns:
        gini.

    """
    if isinstance(data, cp.ndarray):
        new_len = data.shape[1]
    else:
        new_len = len(data.columns)
        data = data.fillna(cp.nan).values.astype(cp.float32)

    if isinstance(empty_slice, cp.ndarray):
        orig_len = empty_slice.shape[1]
    else:
        orig_len = len(empty_slice.columns)
        empty_slice = empty_slice.values

    scores = cp.zeros(new_len)
    len_ratio = int(new_len / orig_len)

    index = cp.arange(new_len, dtype=cp.int32)
    ind = index // len_ratio
    sl = empty_slice[:, ind]

    scores = gini_normalized_gpu_new(data, target, sl)

    if len_ratio != 1:

        scores = scores.reshape((orig_len, len_ratio))
        scores = scores.mean(axis=1)
    return scores


def _get_score_from_pipe_gpu(
    train: GpuDataset,
    target: GpuDataset,
    pipe: Optional[LAMLTransformer] = None,
    empty_slice: Optional[Union[GpuFrame, cp.ndarray]] = None,
) -> cp.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train: gpu Dataset.
        target: gpu Dataset.
        pipe: LAMLTransformer.
        empty_slice: cp.ndarray or gpu DataFrame.

    Returns:
        np.ndarray.

    """

    if pipe is not None:
        train = pipe.fit_transform(train)

    data = train.data
    scores = calc_ginis_gpu(data, target, empty_slice)
    return scores


def rule_based_roles_guess_gpu(stat: cudf.DataFrame) -> Dict[str, ColumnRole]:
    """Create roles dict based on stats.

    Args:
        stat: DataFrame.

    Returns:
        Dict.

    """

    numbers = stat[stat[[x for x in stat.columns if "rule_" in x]].any(axis=1)].copy()
    categories = stat.drop(numbers.index)
    # define encoding types
    roles_dict = {}

    # rules to determinate handling type
    numbers["discrete_rule"] = (~numbers["rule_7"]) & (
        (numbers["binned_scores"] / numbers["raw_scores"]) > 2
    )
    categories["int_rule"] = categories["unique"] < 10
    categories["freq_rule"] = (
        categories["freq_scores"] / categories["encoded_scores"]
    ) > 1.3
    categories["ord_rule"] = categories["unique_rate"] > 0.01

    # numbers with discrete features
    role = NumericRole(np.float32, discretization=True)
    feats = numbers[numbers["discrete_rule"]].to_pandas().index
    roles_dict = {**roles_dict, **{x: role for x in feats}}

    # classic numbers
    role = NumericRole(np.float32)
    feats = numbers[~numbers["discrete_rule"]].to_pandas().index
    roles_dict = {**roles_dict, **{x: role for x in feats}}

    # low cardinal categories
    # role = CategoryRole(np.float32, encoding_type='int')
    feats = categories[categories["int_rule"]].to_pandas().index
    ordinal = categories["ord_rule"][categories["int_rule"]].to_pandas().values
    roles_dict = {
        **roles_dict,
        **{
            x: CategoryRole(np.float32, encoding_type="int", ordinal=y)
            for (x, y) in zip(feats, ordinal)
        },
    }

    # frequency encoded feats
    # role = CategoryRole(np.float32, encoding_type='freq')
    feats = categories[categories["freq_rule"]].to_pandas().index
    ordinal = categories["ord_rule"][categories["freq_rule"]].to_pandas().values
    roles_dict = {
        **roles_dict,
        **{
            x: CategoryRole(np.float32, encoding_type="freq", ordinal=y)
            for (x, y) in zip(feats, ordinal)
        },
    }

    # categories left
    # role = CategoryRole(np.float32)
    feats = (
        categories[(~categories["freq_rule"]) & (~categories["int_rule"])]
        .to_pandas()
        .index
    )
    ordinal = (
        categories["ord_rule"][(~categories["freq_rule"]) & (~categories["int_rule"])]
        .to_pandas()
        .values
    )
    roles_dict = {
        **roles_dict,
        **{
            x: CategoryRole(np.float32, encoding_type="auto", ordinal=y)
            for (x, y) in zip(feats, ordinal)
        },
    }

    return roles_dict


def get_score_from_pipe_gpu(
    train: GpuDataset,
    target: GpuDataset,
    pipe: Optional[LAMLTransformer] = None,
    empty_slice: Optional[GpuFrame] = None,
    n_jobs: int = 1,
) -> cp.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train: gpu Dataset.
        target: gpu Dataset.
        pipe: LAMLTransformer.
        empty_slice: gpu DataFrame.
        n_jobs: int

    Returns:
        np.ndarray.

    """
    if n_jobs == 1:
        return _get_score_from_pipe_gpu(train, target, pipe, empty_slice)

    idx = np.array_split(np.arange(len(train.features)), n_jobs)
    idx = [x for x in idx if len(x) > 0]
    n_jobs = len(idx)

    names = [[train.features[x] for x in y] for y in idx]

    with Parallel(
        n_jobs=n_jobs, prefer="processes", backend="loky", max_nbytes=None
    ) as p:
        res = p(
            delayed(_get_score_from_pipe_gpu)(
                train[:, name], target, pipe, empty_slice[name]
            )
            for name in names
        )
    return cp.concatenate(list(map(cp.array, res)))


def get_numeric_roles_stat_gpu(
    train: GpuDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    manual_roles: Optional[RolesDict] = None,
    n_jobs: int = 1,
) -> cudf.DataFrame:
    """Calculate statistics about different encodings performances.

    We need it to calculate rules about advanced roles guessing.
    Only for numeric data.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: int.
        manual_roles: Dict.
        n_jobs: int.

    Returns:
        DataFrame.

    """
    if manual_roles is None:
        manual_roles = {}

    roles_to_identify = []
    flg_manual_set = []
    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == "Numeric":  # and f != train.target.name:
            roles_to_identify.append(f)
            flg_manual_set.append(f in manual_roles)
    res = cudf.DataFrame(
        columns=[
            "flg_manual",
            "unique",
            "unique_rate",
            "top_freq_values",
            "raw_scores",
            "binned_scores",
            "encoded_scores",
            "freq_scores",
            "nan_rate",
        ],
        index=roles_to_identify,
    )
    res["flg_manual"] = flg_manual_set
    if len(roles_to_identify) == 0:
        return res

    train = train[:, roles_to_identify]
    train_len = train.shape[0]
    if train.folds is None:
        train.folds = set_sklearn_folds(
            train.task, train.target, cv=5, random_state=random_state, group=train.group
        )
    if subsample is not None and subsample < train_len:
        # here need to do the remapping
        # train.data = train.data.sample(subsample, axis=0,
        #                               random_state=random_state)
        idx = cp.random.RandomState(random_state).permutation(train_len)[:subsample]
        train = train[idx]
        train_len = subsample
    target, encoder = get_target_and_encoder_gpu(train)
    empty_slice = train.data.isna()
    # check scores as is

    res["raw_scores"] = get_score_from_pipe_gpu(
        train, target, empty_slice=empty_slice, n_jobs=n_jobs
    )

    # check unique values
    unique_values = None
    top_freq_values = None
    ## transfer memory
    if isinstance(train.data, cudf.DataFrame):

        desc = train.data.nans_to_nulls().astype(object).describe(include="all")
        unique_values = desc.loc["unique"].astype(cp.int32).values[0]
        top_freq_values = desc.loc["freq"].astype(cp.int32).values[0]
    else:
        raise NotImplementedError
    res["unique"] = unique_values
    res["top_freq_values"] = top_freq_values
    res["unique_rate"] = res["unique"] / train_len

    # check binned categorical score
    trf = SequentialTransformer([QuantileBinning_gpu(), encoder()])

    res["binned_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check label encoded scores
    trf = SequentialTransformer(
        [ChangeRoles(CategoryRole(np.float32)), LabelEncoder_gpu(), encoder()]
    )

    res["encoded_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check frequency encoding
    trf = SequentialTransformer(
        [ChangeRoles(CategoryRole(np.float32)), FreqEncoder_gpu()]
    )

    res["freq_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )

    if isinstance(empty_slice, cudf.DataFrame):
        res["nan_rate"] = empty_slice.mean(axis=0).values_host
    else:
        raise NotImplementedError
    return res


def get_category_roles_stat_gpu(
    train: GpuDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    n_jobs: int = 1,
) -> cudf.DataFrame:
    """Search for optimal processing of categorical values.

    Categorical means defined by user or object types.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: seed of random numbers generator.
        n_jobs: number of jobs.

    Returns:
        DataFrame.

    """

    roles_to_identify = []

    dtypes = []

    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == "Category" and role.encoding_type == "auto":
            roles_to_identify.append(f)
            dtypes.append(role.dtype)

    res = cudf.DataFrame(
        columns=["unique", "top_freq_values", "dtype", "encoded_scores", "freq_scores"],
        index=roles_to_identify,
    )
    # res['dtype'] = dtypes

    if len(roles_to_identify) == 0:
        return res, dtypes

    train = train[:, roles_to_identify]
    train_len = train.shape[0]

    if train.folds is None:
        print("No train folds! Assigning...")
        train.folds = set_sklearn_folds(
            train.task, train.target, cv=5, random_state=random_state, group=train.group
        )
    if subsample is not None and subsample < train_len:
        idx = np.random.RandomState(random_state).permutation(train_len)[:subsample]
        train = train[idx]
        # train.data = train.data.sample(subsample, axis=0, random_state=random_state)
        train_len = subsample

    target, encoder = get_target_and_encoder_gpu(train)
    empty_slice = train.data.isna()

    # check label encoded scores
    trf = SequentialTransformer([LabelEncoder_gpu(), encoder()])
    res["encoded_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check frequency encoding
    trf = FreqEncoder_gpu()
    res["freq_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check ordinal encoding
    trf = OrdinalEncoder_gpu()
    res["ord_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    return res, dtypes


def get_null_scores_gpu(
    train: GpuDataset,
    feats: Optional[List[str]] = None,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
) -> pd.Series:
    """Get null scores.

    Args:
        train: Dataset
        feats: list of features.
        subsample: size of subsample.
        random_state: seed of random numbers generator.

    Returns:
        Series.

    """
    if feats is not None:
        train = train[:, feats]

    shape = train.shape

    if subsample is not None and subsample < shape[0]:
        idx = np.random.RandomState(random_state).permutation(shape[0])[:subsample]
        train = train[idx]
        # train.data = train.data.sample(subsample, axis=0,
        #                               random_state=random_state)

    # check task specific
    target, _ = get_target_and_encoder_gpu(train)

    empty_slice = train.data.isnull()
    notnan = empty_slice.sum(axis=0)
    notnan = (notnan != shape[0]) & (notnan != 0)

    notnan_inds = empty_slice.columns[notnan.values_host]
    empty_slice = empty_slice[notnan_inds]

    scores = cp.zeros(shape[1])

    if len(notnan_inds) != 0:
        notnan_inds = np.array(notnan_inds).reshape(-1, 1)
        scores_ = calc_ginis_gpu(empty_slice, target, empty_slice)
        scores[notnan.values_host] = scores_

    res = cudf.Series(scores, index=train.features, name="max_score_2")
    return res
