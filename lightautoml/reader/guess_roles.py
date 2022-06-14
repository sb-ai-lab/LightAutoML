"""Roles guess."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np

from joblib import Parallel
from joblib import delayed
from pandas import DataFrame
from pandas import Series

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.reader.utils import set_sklearn_folds
from lightautoml.transformers.base import ChangeRoles
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.base import SequentialTransformer
from lightautoml.transformers.categorical import FreqEncoder
from lightautoml.transformers.categorical import LabelEncoder
from lightautoml.transformers.categorical import MultiClassTargetEncoder
from lightautoml.transformers.categorical import OrdinalEncoder
from lightautoml.transformers.categorical import TargetEncoder
from lightautoml.transformers.numeric import QuantileBinning


NumpyOrPandas = Union[NumpyDataset, PandasDataset]
RolesDict = Dict[str, ColumnRole]
Encoder = Union[TargetEncoder, MultiClassTargetEncoder]


def ginic(actual: np.ndarray, pred: np.ndarray) -> float:
    """Denormalized gini calculation.

    Args:
        actual: True values.
        pred: Predicted values.

    Returns:
        Metric value.

    """
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    gini_sum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return gini_sum / n


def gini_normalizedc(a: np.ndarray, p: np.ndarray) -> float:
    """Calculated normalized gini.

    Args:
        a: True values.
        p: Predicted values.

    Returns:
        Metric value.

    """
    return ginic(a, p) / ginic(a, a)


def gini_normalized(y_true: np.ndarray, y_pred: np.ndarray, empty_slice: Optional[np.ndarray] = None):
    """Calculate normalized gini index.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        empty_slice: Mask.

    Returns:
        Gini value.

    """
    # TODO: CHECK ABOUT ZERO TARGET SUM
    if empty_slice is None:
        empty_slice = np.isnan(y_pred)
    elif empty_slice.all():
        return 0.0

    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    if empty_slice.ndim > 1:
        empty_slice = empty_slice[:, 0]

    sl = ~empty_slice

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]

    assert (
        y_pred.shape[1] == 1 or y_true.shape[1] == y_pred.shape[1]
    ), "Shape missmatch. Only calculate NxM vs NxM or Nx1 vs NxM"
    outp_size = y_true.shape[1]

    ginis = np.zeros((outp_size,), dtype=np.float32)

    for i in range(outp_size):
        j = min(i, y_pred.shape[1] - 1)

        yt = y_true[:, i][sl]
        yp = y_pred[:, j][sl]
        ginis[i] = gini_normalizedc(yt, yp)

    return np.abs(ginis).mean()


def get_target_and_encoder(train: NumpyOrPandas) -> Tuple[Any, type]:
    """Get target encoder and target based on dataset.

    Args:
        train: Dataset.

    Returns:
        (Target values, Target encoder).

    """
    train = train.empty().to_numpy()
    target = train.target

    if train.task.name == "multiclass":
        n_out = np.max(target) + 1
        target = target[:, np.newaxis] == np.arange(n_out)[np.newaxis, :]
        target = cast(np.ndarray, target).astype(np.float32)
        encoder = MultiClassTargetEncoder
    else:
        encoder = TargetEncoder

    return target, encoder


def calc_ginis(data: np.ndarray, target: np.ndarray, empty_slice: Optional[np.ndarray] = None):
    """

    Args:
        data: np.ndarray.
        target: np.ndarray.
        empty_slice: np.ndarray.

    Returns:
        gini.

    """

    scores = np.zeros(data.shape[1])
    for n in range(data.shape[1]):
        sl = None
        if empty_slice is not None:
            sl = empty_slice[:, n]

        scores[n] = gini_normalized(target, data[:, n], empty_slice=sl)

    return scores


def _get_score_from_pipe(
    train,
    target,
    pipe: Optional[LAMLTransformer] = None,
    empty_slice: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train:  np.ndarray.
        target: np.ndarray.
        pipe: LAMLTransformer.
        empty_slice: np.ndarray.

    Returns:
        np.ndarray.

    """

    shape = train.shape

    if pipe is not None:
        train = pipe.fit_transform(train)

    data = train.data.reshape(shape + (-1,))
    scores = calc_ginis(data, target, empty_slice)

    return scores


def get_score_from_pipe(
    train: NumpyOrPandas,
    target: np.ndarray,
    pipe: Optional[LAMLTransformer] = None,
    empty_slice: Optional[np.ndarray] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train: np.ndarray.
        target: np.ndarray.
        pipe: LAMLTransformer.
        empty_slice: np.ndarray.
        n_jobs: int.

    Returns:
        np.ndarray.

    """

    shape = train.shape
    if n_jobs == 1:
        return _get_score_from_pipe(train, target, pipe, empty_slice)

    idx = np.array_split(np.arange(shape[1]), n_jobs)
    idx = [x for x in idx if len(x) > 0]
    n_jobs = len(idx)
    names = [[train.features[x] for x in y] for y in idx]

    if empty_slice is None:
        empty_slice = [None] * n_jobs
    else:
        empty_slice = [empty_slice[:, x] for x in idx]

    with Parallel(n_jobs=n_jobs, prefer="processes", backend="loky", max_nbytes=None) as p:
        res = p(
            delayed(_get_score_from_pipe)(train[:, name], target, pipe, sl) for (name, sl) in zip(names, empty_slice)
        )
    return np.concatenate(list(map(np.array, res)))


def get_numeric_roles_stat(
    train: NumpyOrPandas,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    manual_roles: Optional[RolesDict] = None,
    n_jobs: int = 1,
) -> DataFrame:
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
        if role.name == "Numeric":
            roles_to_identify.append(f)
            flg_manual_set.append(f in manual_roles)

    res = DataFrame(
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

    train = train[:, roles_to_identify].to_numpy()

    if train.folds is None:
        train.folds = set_sklearn_folds(train.task, train.target, cv=5, random_state=42, group=train.group)

    if subsample is not None:
        idx = np.random.RandomState(random_state).permutation(train.shape[0])[:subsample]
        train = train[idx]

    data, target = train.data, train.target

    # check task specific
    target, encoder = get_target_and_encoder(train)

    # s3d = data.shape + (-1,)
    empty_slice = np.isnan(data)

    # check scores as is
    res["raw_scores"] = get_score_from_pipe(train, target, empty_slice=empty_slice, n_jobs=n_jobs)

    # check unique values
    unique_values = [np.unique(data[:, x][~np.isnan(data[:, x])], return_counts=True) for x in range(data.shape[1])]
    top_freq_values = np.array([max(x[1]) for x in unique_values])
    unique_values = np.array([len(x[0]) for x in unique_values])
    res["unique"] = unique_values
    res["top_freq_values"] = top_freq_values
    res["unique_rate"] = res["unique"] / train.shape[0]

    # check binned categorical score
    trf = SequentialTransformer([QuantileBinning(), encoder()])
    res["binned_scores"] = get_score_from_pipe(train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs)

    # check label encoded scores
    trf = SequentialTransformer([ChangeRoles(CategoryRole(np.float32)), LabelEncoder(), encoder()])
    res["encoded_scores"] = get_score_from_pipe(train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs)

    # check frequency encoding
    trf = SequentialTransformer([ChangeRoles(CategoryRole(np.float32)), FreqEncoder()])
    res["freq_scores"] = get_score_from_pipe(train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs)

    res["nan_rate"] = empty_slice.mean(axis=0)

    return res


def calc_encoding_rules(
    stat: DataFrame,
    numeric_unique_rate: float = 0.999,
    max_to_3rd_rate: float = 1.1,
    binning_enc_rate: float = 2,
    raw_decr_rate: float = 1.1,
    max_score_rate: float = 0.2,
    abs_score_val: float = 0.04,
) -> DataFrame:
    """Calculate rules based on encoding stats.

    Args:
        stat: DataFrame
        numeric_unique_rate: float.
        max_to_3rd_rate: float.
        binning_enc_rate: float.
        raw_decr_rate: float.
        max_score_rate: float.
        abs_score_val: float.

    Returns:
        DataFrame.

    """
    scores_stat = stat[["raw_scores", "binned_scores", "encoded_scores", "freq_scores"]].values

    top_encodings = scores_stat.argsort(axis=1)[:, ::-1]
    sorted_scores = np.take_along_axis(scores_stat, top_encodings, axis=1)

    stat["max_to_3rd_rate"] = sorted_scores[:, 0] / sorted_scores[:, 2]
    stat["max_to_2rd_rate"] = sorted_scores[:, 0] / sorted_scores[:, 1]
    stat["max_score"] = scores_stat.max(axis=1)
    stat["max_score_rate"] = stat["max_score"] / stat["max_score"].max()

    # exact numbers are (my guess)
    # 1 - best score is raw score
    # 2 - top 1 score is binned, top 2 score is raw
    # 3 - 2 unique values
    # 4 - to many unique values
    # 5 - encoding type have no impact
    # 6 - binning encode wins with high rate
    # 7 - raw encoding looses to top with very small rate
    # 8 - feature is weak (lower than abs co and lower than % from max??)
    # 9 - set manually

    # rules

    stat["rule_0"] = top_encodings[:, 0] == 0
    stat["rule_1"] = top_encodings[:, :1].sum(axis=1) == 1
    stat["rule_2"] = stat["unique"] <= 2
    stat["rule_3"] = stat["unique_rate"] > numeric_unique_rate
    stat["rule_4"] = stat["max_to_3rd_rate"] < max_to_3rd_rate
    stat["rule_5"] = (top_encodings[:, 0] == 1) & (stat["max_to_3rd_rate"] > binning_enc_rate)
    stat["rule_6"] = (top_encodings[:, 1] == 0) & (stat["max_to_2rd_rate"] < raw_decr_rate)
    stat["rule_7"] = (stat["max_score_rate"] < max_score_rate) | (stat["max_score"] < abs_score_val)
    stat["rule_8"] = stat["flg_manual"]

    return stat


def rule_based_roles_guess(stat: DataFrame) -> Dict[str, ColumnRole]:
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
    numbers["discrete_rule"] = (~numbers["rule_7"]) & ((numbers["binned_scores"] / numbers["raw_scores"]) > 2)
    categories["int_rule"] = categories["unique"] < 10
    categories["freq_rule"] = (categories["freq_scores"] / categories["encoded_scores"]) > 1.3
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
    # role = CategoryRole(np.float32, encoding_type='int')
    feats = categories[categories["int_rule"]].index
    ordinal = categories["ord_rule"][categories["int_rule"]].values
    roles_dict = {
        **roles_dict,
        **{x: CategoryRole(np.float32, encoding_type="int", ordinal=y) for (x, y) in zip(feats, ordinal)},
    }

    # frequency encoded feats
    # role = CategoryRole(np.float32, encoding_type='freq')
    feats = categories[categories["freq_rule"]].index
    ordinal = categories["ord_rule"][categories["freq_rule"]].values
    roles_dict = {
        **roles_dict,
        **{x: CategoryRole(np.float32, encoding_type="freq", ordinal=y) for (x, y) in zip(feats, ordinal)},
    }

    # categories left
    # role = CategoryRole(np.float32)
    feats = categories[(~categories["freq_rule"]) & (~categories["int_rule"])].index
    ordinal = categories["ord_rule"][(~categories["freq_rule"]) & (~categories["int_rule"])].values
    roles_dict = {
        **roles_dict,
        **{x: CategoryRole(np.float32, encoding_type="auto", ordinal=y) for (x, y) in zip(feats, ordinal)},
    }

    return roles_dict


def get_category_roles_stat(
    train: NumpyOrPandas,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    n_jobs: int = 1,
):
    """Search for optimal processing of categorical values.

    Categorical means defined by user or object types.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: seed of random numbers generator.
        n_jobs: number of jobs.

    Returns:
        result.

    """

    roles_to_identify = []

    dtypes = []

    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == "Category" and role.encoding_type == "auto":
            roles_to_identify.append(f)
            dtypes.append(role.dtype)

    res = DataFrame(
        columns=[
            "unique",
            "top_freq_values",
            "dtype",
            "encoded_scores",
            "freq_scores",
            "ord_scores",
        ],
        index=roles_to_identify,
    )

    res["dtype"] = dtypes

    if len(roles_to_identify) == 0:
        return res

    train = train[:, roles_to_identify].to_pandas()

    if train.folds is None:
        train.folds = set_sklearn_folds(train.task, train.target.values, cv=5, random_state=42, group=train.group)

    if subsample is not None:
        idx = np.random.RandomState(random_state).permutation(train.shape[0])[:subsample]
        train = train[idx]

    # check task specific
    target, encoder = get_target_and_encoder(train)

    empty_slice = train.data.isnull().values

    # check label encoded scores
    trf = SequentialTransformer([LabelEncoder(), encoder()])
    res["encoded_scores"] = get_score_from_pipe(train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs)

    # check frequency encoding
    trf = FreqEncoder()
    res["freq_scores"] = get_score_from_pipe(train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs)

    # check ordinal encoding
    trf = OrdinalEncoder()
    res["ord_scores"] = get_score_from_pipe(train, target, pipe=trf, empty_slice=empty_slice, n_jobs=n_jobs)

    return res


def calc_category_rules(
    stat: DataFrame,
) -> DataFrame:
    """Select best encoding for categories based on stats.

    Args:
        stat: DataFrame.

    Returns:
        DataFrame.

    """
    scores_stat = stat[["encoded_scores", "freq_scores", "ord_scores"]].values

    top_encodings = scores_stat.argsort(axis=1)[:, ::-1]
    stat["max_score"] = scores_stat.max(axis=1)

    # ordinal if
    # 1 - ordinal is top
    # 2 - 2 unique values
    stat["ord_rule_1"] = top_encodings[:, 0] == 2
    stat["ord_rule_2"] = stat["unique"] <= 2
    # freq encoding
    # 1 - freq is top and more than N unique values
    stat["freq_rule_1"] = top_encodings[:, 0] == 1

    # auto if
    # 1 - encoded is top
    stat["auto_rule_1"] = top_encodings[:, 0] == 0

    return stat


def rule_based_cat_handler_guess(stat: DataFrame) -> Dict[str, ColumnRole]:
    """Create roles dict based on stats.

    Args:
        stat: DataFrame.

    Returns:
        Dict.

    """
    # define encoding types
    roles_dict = {}

    # rules to determinate handling type
    freqs = stat[stat[[x for x in stat.columns if "freq_rule_" in x]].any(axis=1)]
    auto = stat[stat[[x for x in stat.columns if "auto_rule_" in x]].any(axis=1)]
    ordinals = stat[stat[[x for x in stat.columns if "ord_rule_" in x]].any(axis=1)]

    for enc_type, st in zip(["freq", "auto", "ord"], [freqs, auto, ordinals]):

        ordinal = False
        if enc_type == "ord":
            enc_type = "auto"
            ordinal = True

        feats = list(st.index)
        dtypes = list(st["dtype"])
        roles_dict = {
            **roles_dict,
            **{x: CategoryRole(dtype=d, encoding_type=enc_type, ordinal=ordinal) for x, d in zip(feats, dtypes)},
        }

    return roles_dict


def get_null_scores(
    train: NumpyOrPandas,
    feats: Optional[List[str]] = None,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
) -> Series:
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
        train = train[:, feats].to_pandas()

    if subsample is not None:
        idx = np.random.RandomState(random_state).permutation(train.shape[0])[:subsample]
        train = train[idx]

    # check task specific
    target, _ = get_target_and_encoder(train)

    empty_slice = train.data.isnull().values
    notnan = empty_slice.sum(axis=0)
    notnan = (notnan != train.shape[0]) & (notnan != 0)

    scores = np.zeros(train.shape[1])
    scores_ = calc_ginis(empty_slice[:, notnan], target, None)
    scores[notnan] = scores_

    res = Series(scores, index=train.features, name="max_score")

    return res
