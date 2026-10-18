import pickle

import numpy as np
import pandas as pd
import pytest

from scipy.stats import spearmanr
from sklearn import metrics as skm

import lightautoml.utils.logging  # noqa: F401

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.tasks import Task
from lightautoml.tasks.metrics import _METRICS
from lightautoml.tasks.metrics import get_metric
from lightautoml.tasks.metrics.classification import expected_calibration_error
from lightautoml.tasks.metrics.classification import roc_auc_with_nans
from lightautoml.tasks.metrics.ranking import average_precision_at_k
from lightautoml.tasks.metrics.ranking import average_precision_in_top_k
from lightautoml.tasks.metrics.ranking import mean_average_precision_at_k
from lightautoml.tasks.metrics.ranking import mean_reciprocal_rank
from lightautoml.tasks.metrics.ranking import ndcg_at_k
from lightautoml.tasks.metrics.ranking import precision_at_k
from lightautoml.tasks.metrics.ranking import recall_at_k
from lightautoml.tasks.metrics.regression import adjusted_r2
from lightautoml.tasks.metrics.regression import mae_to_mean
from lightautoml.tasks.metrics.regression import mape
from lightautoml.tasks.metrics.regression import mape_without_zeros
from lightautoml.tasks.metrics.regression import median_absolute_percentage_error
from lightautoml.tasks.metrics.regression import one_minus_wape
from lightautoml.tasks.metrics.regression import powerstat
from lightautoml.tasks.metrics.regression import rmse
from lightautoml.tasks.metrics.regression import smape
from lightautoml.tasks.metrics.regression import spearman


@pytest.mark.parametrize(
    "name,reference,greater",
    [
        ("pr_auc", skm.average_precision_score, True),
        ("logloss", skm.log_loss, False),
        ("brier_loss", skm.brier_score_loss, False),
        ("roc_auc_score", skm.roc_auc_score, True),
    ],
)
def test_probability_metrics_keep_probabilities_and_weights(name, reference, greater):
    target = np.array([0, 1, 1, 0])
    pred = np.array([0.45, 0.55, 0.8, 0.2])
    weights = np.array([1, 3, 2, 4])
    expected = reference(target, pred, sample_weight=weights)
    task = Task("binary", metric=name)
    assert task.greater_is_better == greater
    assert task.metric_func(target, pred, sample_weight=weights) == pytest.approx(expected)
    dataset = NumpyDataset(pred[:, None], ["prediction"], NumericRole(), task, target=target, weights=weights)
    assert task.get_dataset_metric()(dataset) == pytest.approx(expected if greater else -expected)


@pytest.mark.parametrize(
    "name,reference",
    [("f1_score", skm.f1_score), ("precision_score", skm.precision_score), ("recall_score", skm.recall_score)],
)
def test_label_metrics_cutoff_and_weighting(name, reference):
    target, pred, weights = [0, 1, 1, 0], np.array([0.5, 0.5, 0.8, 0.9]), [1, 2, 3, 4]
    metric = get_metric(name, cutoff=0.5)
    assert metric(target, pred, sample_weight=weights) == pytest.approx(
        reference(target, pred >= 0.5, sample_weight=weights)
    )
    assert get_metric(name, cutoff=0.7)(target, pred, sample_weight=weights) == pytest.approx(
        reference(target, pred >= 0.7, sample_weight=weights)
    )
    assert get_metric(name)([0, 0], [0.1, 0.2]) == 0


def test_multiclass_missing_predicted_class_does_not_reencode_labels():
    target = np.array([0, 1, 2, 2])
    pred = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.1, 0.6, 0.3]])
    assert get_metric("accuracy_score", task="multiclass")(target, pred) == 0.5
    assert Task("multiclass", metric="f1_macro", metric_params={"zero_division": 0}).metric_func(
        target, pred
    ) == pytest.approx(skm.f1_score(target, pred.argmax(axis=1), average="macro", zero_division=0))


def test_fractional_fbeta_and_multilabel_hamming_accuracy():
    target, pred = np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.8, 0.9])
    assert get_metric("fbeta_score", beta=0.5)(target, pred) == pytest.approx(
        skm.fbeta_score(target, pred >= 0.5, beta=0.5)
    )
    metric = get_metric("hamming_accuracy", task="multilabel")
    assert metric([[1, 0], [0, 1]], [[0.9, 0.2], [0.1, 0.8]]) == 1
    assert metric([[1, 0], [0, 1]], [[0.1, 0.9], [0.8, 0.2]]) == 0


def test_ece_bins_include_endpoints_and_use_probability_scores():
    # First bin has residual 1, last bin residual -1: absolute residuals sum to 2.
    assert expected_calibration_error([1, 0], [0, 1]) == 1
    assert expected_calibration_error([0, 1], [0, 1]) == 0
    assert expected_calibration_error([0, 1], [0.2, 0.6], [1, 3], n_bins=2) == pytest.approx(0.35)
    # Exactly .5 belongs to the first bin, so residuals cancel within it.
    assert expected_calibration_error([0, 1], [0.5, 0.5], n_bins=2) == 0
    assert expected_calibration_error([0, 1], [[0.8, 0.2], [0.7, 0.3]], n_bins=1) == pytest.approx(0.25)
    assert get_metric("ece", task="multilabel")([[1, 0], [0, 1]], [[0, 1], [1, 0]]) == 1


@pytest.mark.parametrize("kwargs", [{"n_bins": 0}, {"n_bins": True}, {"n_bins": 1.5}])
def test_ece_rejects_invalid_bins(kwargs):
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.1, 0.9], **kwargs)


def test_partial_roc_metrics():
    target, pred, weights = [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], [1, 2, 3, 4]
    assert get_metric("rec@spec", threshold=1)(target, pred, sample_weight=weights) == pytest.approx(4 / 7)
    assert get_metric("aucroc@fpr", threshold=0.2)(target, pred, sample_weight=weights) == pytest.approx(
        skm.roc_auc_score(target, pred, sample_weight=weights, max_fpr=0.2)
    )
    with pytest.raises(ValueError, match="both binary classes"):
        get_metric("rec@spec")([1, 1], [0.2, 0.8])


def test_nan_aware_auc_keeps_weight_alignment_and_ignores_degenerate_columns():
    target = pd.DataFrame([[0, 1, np.nan], [1, 1, 0], [np.nan, 1, 1], [1, 1, 0]], index=[8, 2, 5, 9])
    pred = pd.DataFrame([[0.2, 0.5, 0.4], [0.7, 0.5, 0.6], [0.9, 0.5, 0.8], [0.1, 0.5, 0.7]], index=target.index)
    weights = np.array([1, 2, 5, 3])
    expected = (skm.roc_auc_score([0, 1, 1], [0.2, 0.7, 0.1], sample_weight=weights[[0, 1, 3]]) + 1) / 2
    assert roc_auc_with_nans(target, pred, weights) == pytest.approx(expected)
    assert get_metric("gini_with_nans", task="multilabel")(target, pred, weights) == pytest.approx(
        (2 * expected - 1) * 100
    )
    assert get_metric("roc_auc_score_average_samples", task="multilabel")(target, pred, weights) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="No target column"):
        roc_auc_with_nans([[1, np.nan], [1, np.nan]], [[0.2, 0.3], [0.8, 0.1]])


def test_percentage_errors_and_zero_alignment():
    assert smape([0, 2, -2], [0, 0, 2]) == pytest.approx(400 / 3)
    assert mape_without_zeros([0, 2, 0, 4], [99, 1, 99, 3]) == 0.375
    assert mape_without_zeros([0, 2, 0, 4], [99, 1, 99, 3], [100, 1, 100, 3]) == 0.3125
    assert mape([0, 2], [0, 1]) == 0.25
    assert median_absolute_percentage_error([1, 2, 4], [1, 1, 3]) == 25
    # Absolute denominator works for signed targets.
    assert one_minus_wape([-2, 2], [-1, 1]) == 0.5
    with pytest.raises(ValueError, match="nonzero target"):
        mape_without_zeros([0, 0], [1, 1])
    with pytest.raises(ValueError, match="positive target mean"):
        one_minus_wape([0, 0], [0, 0])
    with pytest.raises(ValueError, match="positive target mean"):
        mae_to_mean([-1, 1], [0, 0])


def test_multioutput_regression_aggregates_outputs_after_rows():
    target = np.array([[1, 10], [2, 20], [3, 40]])
    pred = np.array([[1, 7], [4, 24], [2, 35]])
    weights = np.array([1, 2, 4])
    expected = np.mean([np.sqrt(np.average((t - p) ** 2, weights=weights)) for t, p in zip(target.T, pred.T)])
    assert rmse(target, pred, weights) == pytest.approx(expected)
    assert smape(target, pred, weights) == pytest.approx(
        np.mean([smape(t, p, weights) for t, p in zip(target.T, pred.T)])
    )


def test_adjusted_r2_requires_feature_count_and_enough_rows():
    target, pred = np.arange(8), np.arange(8) + 0.5
    assert adjusted_r2(target, pred, n_features=2) == pytest.approx(1 - (1 - skm.r2_score(target, pred)) * 7 / 5)
    with pytest.raises(ValueError, match="Required parameters"):
        Task("reg", metric="r2_adj")
    with pytest.raises(ValueError, match="more than"):
        adjusted_r2(target[:3], pred[:3], n_features=2)


def test_spearman_and_powerstat_reference_cases():
    target, pred = [1, 3, 2, 4], [4, 1, 1, 2]
    assert spearman(target, pred) == pytest.approx(spearmanr(target, pred)[0])
    assert powerstat([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(100)
    assert powerstat([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-100)
    assert powerstat([0, 1, 2, 3], [1, 1, 1, 1]) == pytest.approx(0)
    with pytest.raises(ValueError, match="constant"):
        spearman([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="constant"):
        powerstat([1, 1, 1], [1, 2, 3])


def test_top_k_metrics_and_two_ap_definitions():
    target, pred = [1, 0, 1, 1], [0.9, 0.8, 0.7, 0.6]
    assert precision_at_k(target, pred, k=2) == 0.5
    assert recall_at_k(target, pred, k=2) == pytest.approx(1 / 3)
    assert average_precision_at_k(target, pred, k=2) == pytest.approx(1 / 3)
    assert average_precision_in_top_k(target, pred, k=2) == 1
    assert average_precision_at_k(target, pred, k=100) == pytest.approx(skm.average_precision_score(target, pred))
    assert precision_at_k(target, pred, k=0.5) == 0.5
    assert precision_at_k(target, pred, k=0.01) == 1
    assert precision_at_k([1, 0], [0.5, 0.5], k=1) == 1
    assert recall_at_k([0, 0], [0.2, 0.8]) == 0
    assert average_precision_in_top_k([0, 0], [0.2, 0.8]) == 0
    assert precision_at_k(target, pred, [0, 1, 2, 3], k=1) == 0


@pytest.mark.parametrize("k", [0, -1, True, 1.5, np.nan, "3"])
def test_top_k_rejects_ambiguous_or_invalid_k(k):
    with pytest.raises(ValueError, match="k must"):
        precision_at_k([0, 1], [0.1, 0.9], k=k)


def test_query_ranking_and_item_id_mrr():
    target = np.array([[1, 0, 1], [0, 1, 0]])
    pred = np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
    assert mean_average_precision_at_k(target, pred, [1, 3], k=2) == 0.5
    assert ndcg_at_k(target, pred, [1, 3], k=2) == pytest.approx(
        skm.ndcg_score(target, pred, sample_weight=[1, 3], k=2)
    )
    assert get_metric("mrr", task="multilabel")(target, pred, sample_weight=[1, 3]) == 0.625
    assert mean_reciprocal_rank([["b"], ["x"]], [["a", "b"], ["y", "z"]]) == 0.25
    assert get_metric("mrr", task="ranking")([[2], [1]], [[3, 2], [1, 2]]) == 0.75


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 1], [1], [[1], [2]]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        smape([1, 2], [2, 3], weights)


def test_parameter_errors_and_serialization():
    with pytest.raises(ValueError, match="Unexpected parameters"):
        Task("binary", metric="pr_auc", metric_params={"cutoff": 0.5})
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("f1_score", task="reg")
    metric = pickle.loads(pickle.dumps(get_metric("precision@k", k=1)))
    assert metric.greater_is_better
    assert metric([0, 1], [0.1, 0.9]) == 1


@pytest.mark.parametrize(
    "task_name,name", [(t, n) for t in ["binary", "reg", "multiclass", "multi:reg", "multilabel"] for n in _METRICS[t]]
)
def test_registered_task_metrics_evaluate_real_shapes(task_name, name):
    rng = np.random.RandomState(42)
    if task_name == "binary":
        target, pred = np.tile([0, 1], 15), rng.uniform(0.1, 0.9, 30)
    elif task_name == "multiclass":
        target = np.tile([0, 1, 2], 10)
        pred = rng.dirichlet([1, 1, 1], 30)
    elif task_name == "multilabel":
        target = np.tile([[0, 1, 1], [1, 0, 0]], (15, 1))
        pred = rng.uniform(0.1, 0.9, (30, 3))
    else:
        target = rng.uniform(1, 3, (30, 2) if task_name == "multi:reg" else 30)
        pred = target + rng.uniform(0, 0.5, target.shape)
    params = (
        {"beta": 0.5}
        if name in ("fbeta", "fbeta_score")
        else {"n_features": 2}
        if name in ("r2_adj", "adjusted_r2")
        else {}
    )
    task = Task(task_name, metric=name, metric_params=params)
    value = task.metric_func(target, pred)
    assert np.isscalar(value) and np.isfinite(value)
    assert task.greater_is_better == _METRICS[task_name][name].greater_is_better
    if task_name == "binary":
        assert task.losses["torch"].metric_func(target, pred[:, None]) == pytest.approx(
            value if task.greater_is_better else -value
        )
