import abc

from typing import Tuple

import numpy as np

from sklearn.metrics import auc
from sklearn.utils.multiclass import type_of_target


_available_uplift_modes = ("qini", "cum_gain", "adj_qini")


class ConstPredictError(Exception):
    pass


class TUpliftMetric(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def __call__(self, y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray) -> float:
        pass


def perfect_uplift_curve(y_true: np.ndarray, treatment: np.ndarray):
    """Calculate perfect curve

    Method return curve's coordinates if the model is a perfect.
    Perfect model ranking:
        If type if 'y_true' is 'binary':
            1) Treatment = 1, Target = 1
            2) Treatment = 0, Target = 0
            3) Treatment = 1, Target = 0
            4) Treatment = 0, Target = 1

        If type if 'y_true' is 'continuous':
            Not implemented

    Args:
        y_true: Target values
        treatment: Treatment column

    Returns:
        perfect curve

    """
    if type_of_target(y_true) == "continuous" and np.any(y_true < 0.0):
        raise Exception("For a continuous target, the perfect curve is only available for non-negative values")

    if type_of_target(y_true) == "binary":
        perfect_control_score = (treatment == 0).astype(int) * (2 * (y_true != 1).astype(int) - 1)
        perfect_treatment_score = (treatment == 1).astype(int) * 2 * (y_true == 1).astype(int)
        perfect_uplift = perfect_treatment_score + perfect_control_score
    elif type_of_target(y_true) == "continuous":
        raise NotImplementedError("Can't calculate perfect curve for continuous target")
    else:
        raise RuntimeError("Only 'binary' and 'continuous' targets are available")

    return perfect_uplift


def _get_uplift_curve(
    y_treatment: np.ndarray,
    y_control: np.ndarray,
    n_treatment: np.ndarray,
    n_control: np.ndarray,
    mode: str,
):
    """Calculate uplift curve

    Args:
        y_treatment: Cumulative number of target in treatment group
        y_control: Cumulative number of target in control group
        num_treatment: Cumulative number of treatment group
        num_control: Cumulative number of treatment group
        mode: Name of available metrics

    Returns:
        curve for current mode

    """
    assert mode in _available_uplift_modes, "Mode isn't available"

    if mode == "qini":
        curve_values = y_treatment / n_treatment[-1] - y_control / n_control[-1]
    elif mode == "cum_gain":
        treatment_target_rate = np.nan_to_num(y_treatment / n_treatment, 0.0)
        control_target_rate = np.nan_to_num(y_control / n_control, 0.0)
        curve_values = treatment_target_rate - control_target_rate
        n_join = n_treatment + n_control
        curve_values = curve_values * n_join / n_join[-1]
    elif mode == "adj_qini":
        normed_factor = np.nan_to_num(n_treatment / n_control, 0.0)
        normed_y_control = y_control * normed_factor
        curve_values = (y_treatment - normed_y_control) / n_treatment[-1]

    return curve_values


def calculate_graphic_uplift_curve(
    y_true: np.ndarray,
    uplift_pred: np.ndarray,
    treatment: np.ndarray,
    mode: str = "adj_qini",
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate uplift curve

    Args:
        y_true: Target values
        uplift: Prediction of models
        treatment: Treatment column
        mode: Name of available metrics

    Returns:
        xs, ys - curve's coordinates

    """
    # assert not np.all(uplift_pred == uplift_pred[0]), "Can't calculate uplift curve for constant predicts"
    if np.all(uplift_pred == uplift_pred[0]):
        raise ConstPredictError("Can't calculate uplift curve for constant predicts")

    if type_of_target(y_true) == "continuous" and np.any(y_true < 0.0):
        raise Exception("For a continuous target, the perfect curve is only available for non-negative values")

    sorted_indexes = np.argsort(uplift_pred)[::-1]
    y_true, uplift_pred, treatment = (
        y_true[sorted_indexes],
        uplift_pred[sorted_indexes],
        treatment[sorted_indexes],
    )

    indexes = np.where(np.diff(uplift_pred))[0]
    indexes = np.insert(indexes, indexes.size, uplift_pred.shape[0] - 1)

    n_treatment_samples_cs = np.cumsum(treatment)[indexes].astype(np.int64)
    n_join_samples_cs = indexes + 1
    n_control_samples_cs = n_join_samples_cs - n_treatment_samples_cs

    y_true_control, y_true_treatment = y_true.copy(), y_true.copy()
    y_true_control[treatment == 1] = 0
    y_true_treatment[treatment == 0] = 0

    y_true_control_cs = np.cumsum(y_true_control)[indexes]
    y_true_treatment_cs = np.cumsum(y_true_treatment)[indexes]

    curve_values = _get_uplift_curve(
        y_true_treatment_cs,
        y_true_control_cs,
        n_treatment_samples_cs,
        n_control_samples_cs,
        mode,
    )

    n_join_samples = np.insert(n_join_samples_cs, 0, 0)
    curve_values = np.insert(curve_values, 0, 0)
    rate_join_samples = n_join_samples / n_join_samples[-1]

    return rate_join_samples, curve_values


def calculate_uplift_auc(
    y_true: np.ndarray,
    uplift_pred: np.ndarray,
    treatment: np.ndarray,
    mode: str = "adj_qini",
    normed: bool = False,
):
    """Calculate area under uplift curve

    Args:
        y_true: Target values
        uplift_pred: Prediction of meta model
        treatment: Treatment column
        mode: Name of available metrics
        normed: Normed AUC: (AUC - MIN_AUC) / (MAX_AUC - MIN_AUC)

    Returns:
        auc_score: Area under model uplift curve

    """
    xs, ys = calculate_graphic_uplift_curve(y_true, uplift_pred, treatment, mode)

    uplift_auc = auc(xs, ys)

    if normed:
        min_auc, max_auc = calculate_min_max_uplift_auc(y_true, treatment, mode)

        uplift_auc = (uplift_auc - min_auc) / (max_auc - min_auc)

    return uplift_auc


def calculate_min_max_uplift_auc(y_true: np.ndarray, treatment: np.ndarray, mode: str = "adj_qini"):
    """Calculate AUC uplift curve for `base` and `perfect` models

    Args:
        y_true: Target values
        treatment: Treatment column
        mode: Name of available metrics

    Returns:
        auc_base: Area under `base`.
        auc_perfect: Area under `perfect` model curve

    """
    diff_target_rate = y_true[treatment == 1].mean() - y_true[treatment == 0].mean()
    xs_base, ys_base = np.array([0, 1]), np.array([0, diff_target_rate])

    perfect_uplift = perfect_uplift_curve(y_true, treatment)
    xs_perfect, ys_perfect = calculate_graphic_uplift_curve(y_true, perfect_uplift, treatment, mode)

    auc_base = auc(xs_base, ys_base)
    auc_perfect = auc(xs_perfect, ys_perfect)

    return auc_base, auc_perfect


def calculate_uplift_at_top(y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray, top: float = 30):
    """Calculate Uplift metric at TOP

    Calculate uplift metric at top

    Args:
        y_true: Target values
        uplift_pred: Prediction of meta model
        treatment: Treatment column
        top: Rate, value between (0, 100]

    Returns:
        score: Score

    """
    # assert not np.all(uplift_pred == uplift_pred[0]), "Can't calculate for constant predicts."

    uplift_percentile = np.percentile(uplift_pred, 100 - top)
    mask_top = uplift_pred > uplift_percentile

    control_true_top = y_true[(treatment == 0) & mask_top].sum()
    treatment_true_top = y_true[(treatment == 1) & mask_top].sum()

    n_control_samples = (treatment[mask_top] == 0).sum()
    n_treatment_samples = (treatment[mask_top] == 1).sum()

    mean_control_value = control_true_top / n_control_samples if n_control_samples > 0 else 0.0
    mean_treatment_value = treatment_true_top / n_treatment_samples if n_treatment_samples > 0 else 0.0

    score = mean_treatment_value - mean_control_value

    return score


def calculate_total_score(y_true: np.ndarray, uplift_pred: np.ndarray, treatment: np.ndarray, top: float = 30):
    """Calculate total target

    Args:
        y_true: Target values
        uplift_pred: Prediction of meta model
        treatment: Treatment column
        top: Rate, value between (0, 100]

    Returns:
        score: Score

    """
    uplift_percentile = np.percentile(uplift_pred, 100 - top)
    mask_top = uplift_pred > uplift_percentile
    mask_treatment = treatment == 1
    treatment_rate = mask_treatment.mean()

    control_true_top = y_true[(~mask_treatment) & (~mask_top)].mean()
    treatment_true_top = y_true[mask_treatment & mask_top].mean()

    score = control_true_top * (1 - treatment_rate) + treatment_true_top * treatment_rate

    return score
