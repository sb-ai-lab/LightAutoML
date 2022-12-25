"""Roles guess on GPU."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

from copy import deepcopy

import torch
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lightautoml_gpu.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml_gpu.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml_gpu.dataset.roles import CategoryRole
from lightautoml_gpu.dataset.roles import ColumnRole
from lightautoml_gpu.dataset.roles import NumericRole
from lightautoml_gpu.reader.utils import set_sklearn_folds
from lightautoml_gpu.transformers.base import ChangeRoles
from lightautoml_gpu.transformers.base import LAMLTransformer
from lightautoml_gpu.transformers.base import SequentialTransformer

from lightautoml_gpu.transformers.gpu.categorical_gpu import FreqEncoderGPU
from lightautoml_gpu.transformers.gpu.categorical_gpu import LabelEncoderGPU
from lightautoml_gpu.transformers.gpu.categorical_gpu import MultiClassTargetEncoderGPU
from lightautoml_gpu.transformers.gpu.categorical_gpu import OrdinalEncoderGPU
from lightautoml_gpu.transformers.gpu.categorical_gpu import TargetEncoderGPU
from lightautoml_gpu.transformers.gpu.categorical_gpu import MultioutputTargetEncoderGPU

from lightautoml_gpu.transformers.gpu.numeric_gpu import QuantileBinningGPU

RolesDict = Dict[str, ColumnRole]
EncoderGPU = Union[TargetEncoderGPU, MultiClassTargetEncoderGPU, MultioutputTargetEncoderGPU]
GpuFrame = Union[cudf.DataFrame]
GpuDataset = Union[CudfDataset, CupyDataset]


def ginic_gpu(actual: GpuFrame, pred: GpuFrame, empty_slice) -> float:
    """Denormalized gini calculation.

    Args:
        actual_pred: array with true and predicted values
        inds: list of indices for true and predicted values

    Returns:
        Metric value

    """

    actual[empty_slice] = 0
    pred[empty_slice] = 0
    n = cp.sum(~empty_slice, axis=0)

    ids = cp.argsort(pred, axis=0)
    a_s = cp.take_along_axis(actual, ids, axis=0)

    a_c = a_s.cumsum(axis=0)

    a_c[cp.take_along_axis(empty_slice, ids, axis=0)] = 0
    gini_sum = a_c.sum(axis=0) / a_s.sum(axis=0) - (n + 1) / 2.0
    return gini_sum / n


def gini_normalizedc_gpu(a: GpuFrame, p: GpuFrame, empty_slice) -> float:
    """Calculated normalized gini.

    Args:
        a_p: array with true and predicted values

    Returns:
        Metric value.

    """

    out = ginic_gpu(a, p, empty_slice) / ginic_gpu(a, a, empty_slice)

    # assert not cp.isnan(out), 'gini index is givin nan, is that ok? {0} and {1}'.format(a, p)
    return out


def gini_normalized_gpu(
    y: GpuFrame, target: GpuFrame, sl: GpuFrame = None
) -> float:
    """Calculate normalized gini index for dataframe data.

    Args:
        y: data.
        true_cols: columns with true data.
        pred_cols: columns with predict data.
        sl: Mask.

    Returns:
        Gini value.

    """
    if sl is None:
        sl = cp.isnan(y)
    all_true = sl.all()
    if all_true:
        return 0.0

    sl = sl.reshape(sl.shape[0], -1)

    outp_size = 1 if target.ndim <= 1 else target.shape[1]
    pred_size = 1 if y.ndim <= 1 else y.shape[1]

    index_i = np.arange(pred_size, dtype=np.int32)
    index_i = np.repeat(index_i, outp_size)
    index_j = np.arange(outp_size)
    index_j = np.repeat([index_j], pred_size, axis=0).reshape(-1)
    y = y.reshape(y.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    row_col_const = 200000000
    batch_size = row_col_const // target.shape[0]
    ginis_new = []
    for i in range((index_j.shape[0] // batch_size) + 1):
        end = min((i + 1) * batch_size, index_j.shape[0])
        ginis_new.append(
            gini_normalizedc_gpu(
                target[:, index_j[i * batch_size : end]],
                y[:, index_i[i * batch_size : end]],
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
    if isinstance(target, (cudf.Series, cudf.DataFrame)):
        target = target.values

    if train.task.name == "multiclass":
        n_out = cp.max(target) + 1
        target = target[:, cp.newaxis] == cp.arange(n_out)[cp.newaxis, :]
        encoder = MultiClassTargetEncoderGPU
    elif (train.task.name == "multi:reg") or (train.task.name == "multilabel"):
        target = cast(cp.ndarray, target).astype(cp.float32)
        encoder = MultioutputTargetEncoderGPU
    else:
        encoder = TargetEncoderGPU
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

    ind = cp.arange(new_len, dtype=cp.int32) // len_ratio

    scores = gini_normalized_gpu(data, target, empty_slice[:, ind])

    if len_ratio != 1:

        scores = scores.reshape((orig_len, len_ratio))
        scores = scores.mean(axis=1)
    return scores

from lightautoml_gpu.reader.guess_roles import calc_ginis

def _get_score_from_pipe_gpu(
    train: GpuDataset,
    target: GpuDataset,
    pipe: Optional[LAMLTransformer] = None,
    empty_slice: Optional[Union[GpuFrame, cp.ndarray]] = None,
    dev_id: int = None
) -> cp.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train: gpu Dataset.
        target: gpu Dataset.
        pipe: LAMLTransformer.
        empty_slice: cp.ndarray or gpu DataFrame.
        dev_id: gpu device id for parallel work
    Returns:
        cp.ndarray.

    """
    if dev_id is not None:
        cp.cuda.runtime.setDevice(dev_id)
        if isinstance(train.data, cp.ndarray):
            train.set_data(cp.copy(train.data), train.features,
                           train.roles)
        else:
            train.set_data(train.data.copy(), None, train.roles)
        target = cp.copy(target)
        if empty_slice is not None:
            if isinstance(empty_slice, cp.ndarray):
                empty_slice = cp.copy(empty_slice)
            else:
                empty_slice = empty_slice.copy()
    if pipe is not None:
        train = pipe.fit_transform(train)

    train = train.to_numpy()
    data = train.data
    new_len = data.shape[1]

    if isinstance(empty_slice, cp.ndarray):
        orig_len = empty_slice.shape[1]
    else:
        orig_len = len(empty_slice.columns)
        empty_slice = empty_slice.values

    empty_slice = cp.asnumpy(empty_slice)
    len_ratio = int(new_len / orig_len)
    target = cp.asnumpy(target)
    data = data.reshape((data.shape[0], orig_len, len_ratio))
    
    scores = calc_ginis(data, target, empty_slice)
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
    feats = numbers[numbers["discrete_rule"]].index
    roles_dict = {**roles_dict, **{x: role for x in feats}}

    # classic numbers
    role = NumericRole(np.float32)
    feats = numbers[~numbers["discrete_rule"]].index
    roles_dict = {**roles_dict, **{x: role for x in feats}}

    # low cardinal categories
    feats = categories[categories["int_rule"]].index
    ordinal = categories["ord_rule"][categories["int_rule"]].index
    roles_dict = {
        **roles_dict,
        **{
            x: CategoryRole(np.float32, encoding_type="int", ordinal=y)
            for (x, y) in zip(feats, ordinal)
        },
    }

    # frequency encoded feats
    feats = categories[categories["freq_rule"]].index
    ordinal = categories["ord_rule"][categories["freq_rule"]].index
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
        .index
    )
    ordinal = (
        categories["ord_rule"][(~categories["freq_rule"]) & (~categories["int_rule"])]
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

    n_gpus = torch.cuda.device_count()
    ids = [i % n_gpus for i in range(n_jobs)]
    names = [[train.features[x] for x in y] for y in idx]
    pipes = [deepcopy(pipe) for i in range(n_jobs)]
    with Parallel(n_jobs=n_jobs, prefer="threads") as p:
        res = p(delayed(_get_score_from_pipe_gpu)(train[:, names[i]], target, pipes[i], empty_slice[names[i]], ids[i]) for i in range(n_jobs))
    return np.concatenate(list(map(np.array, res)))


def get_numeric_roles_stat_gpu(
    train: GpuDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    manual_roles: Optional[RolesDict] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
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
    res = pd.DataFrame(
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
        idx = np.random.RandomState(random_state).permutation(train_len)[:subsample]
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
    # transfer memory
    if isinstance(train.data, cudf.DataFrame):

        desc = train.data.nans_to_nulls().astype(object).describe(include="all")
        unique_values = cp.asnumpy(desc.loc["unique"].astype(cp.int32).values[0])
        top_freq_values = cp.asnumpy(desc.loc["freq"].astype(cp.int32).values[0])
        desc = None
    else:
        raise NotImplementedError
    res["unique"] = unique_values
    res["top_freq_values"] = top_freq_values
    res["unique_rate"] = res["unique"] / train_len

    # check binned categorical score
    trf = SequentialTransformer([QuantileBinningGPU(), encoder()])
    res["binned_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check label encoded scores
    trf = SequentialTransformer(
        [ChangeRoles(CategoryRole(np.float32)), LabelEncoderGPU(), encoder()]
    )

    res["encoded_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check frequency encoding
    trf = SequentialTransformer(
        [ChangeRoles(CategoryRole(np.float32)), FreqEncoderGPU()]
    )

    res["freq_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    if isinstance(empty_slice, cudf.DataFrame):
        res["nan_rate"] = empty_slice.mean(axis=0).values_host
    else:
        raise NotImplementedError
    res = res.fillna(np.nan)
    return res


def get_category_roles_stat_gpu(
    train: GpuDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
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

    res = pd.DataFrame(
        columns=["unique", "top_freq_values", "dtype", "encoded_scores", "freq_scores"],
        index=roles_to_identify,
    )
    # res['dtype'] = dtypes

    if len(roles_to_identify) == 0:
        return res, dtypes

    train = train[:, roles_to_identify]
    train_len = train.shape[0]

    if train.folds is None:
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
    trf = SequentialTransformer([LabelEncoderGPU(), encoder()])
    res["encoded_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check frequency encoding
    trf = FreqEncoderGPU()
    res["freq_scores"] = get_score_from_pipe_gpu(
        train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs
    )
    # check ordinal encoding
    trf = OrdinalEncoderGPU()
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
    res = pd.Series(cp.asnumpy(scores), index=train.features, name="max_score_2")
    return res
