"""Named evaluation metrics and their optimization direction.

Use ``Task(..., metric=name, metric_params={...})`` for model selection, or
``get_metric(name, task=...)(target, prediction)`` for standalone evaluation.
Ranking matrices contain one query per row. Uplift and rolling-forecast
metrics are evaluation helpers, not additional training task types.
"""

from functools import partial

from sklearn import metrics as skm

from .classification import expected_calibration_error
from .classification import gini
from .classification import gini_with_nans
from .classification import hamming_accuracy
from .classification import label_metric
from .classification import multilabel_brier_loss
from .classification import multilabel_ece
from .classification import partial_auc
from .classification import recall_at_specificity
from .classification import roc_auc_with_nans
from .forecast import mean_mae
from .forecast import mean_mse
from .ranking import average_precision_at_k
from .ranking import average_precision_in_top_k
from .ranking import mean_average_precision_at_k
from .ranking import mean_reciprocal_rank
from .ranking import ndcg_at_k
from .ranking import precision_at_k
from .ranking import recall_at_k
from .ranking import reciprocal_rank_score
from .regression import adjusted_r2
from .regression import mae_to_mean
from .regression import mae_to_module_mean
from .regression import mape
from .regression import mape_without_zeros
from .regression import median_absolute_percentage_error
from .regression import one_minus_wape
from .regression import powerstat
from .regression import rmse
from .regression import smape
from .regression import spearman
from .uplift import qini_auc_score
from .uplift import uplift_at_k
from .uplift import uplift_auc_score
from .uplift import weighted_average_uplift


class _MetricSpec:
    """Internal callable, direction and accepted parameter metadata."""

    def __init__(self, func, greater_is_better, parameters=(), required=()):
        self.func = func
        self.greater_is_better = greater_is_better
        self.parameters = set(parameters)
        self.required = set(required)

    def validate(self, name, parameters):
        extra = set(parameters) - self.parameters
        missing = self.required - set(parameters)
        if extra:
            raise ValueError("Unexpected parameters for {}: {}".format(name, sorted(extra)))
        if missing:
            raise ValueError("Required parameters for {}: {}".format(name, sorted(missing)))


_METRICS = {
    task: {} for task in ("binary", "multiclass", "multilabel", "reg", "multi:reg", "ranking", "uplift", "timeseries")
}


def _register(tasks, names, func, greater_is_better, parameters=(), required=()):
    spec = _MetricSpec(func, greater_is_better, parameters, required)
    for task in tasks:
        for name in names:
            _METRICS[task][name] = spec


for _task in ("binary", "multiclass", "multilabel"):
    _multiclass = _task == "multiclass"
    _register(
        [_task],
        ["accuracy_score"],
        partial(label_metric, metric=skm.accuracy_score, multiclass=_multiclass),
        True,
        ["cutoff"],
    )
    for _name, _function in (
        ("f1", skm.f1_score),
        ("precision", skm.precision_score),
        ("recall", skm.recall_score),
        ("jaccard", skm.jaccard_score),
    ):
        _averages = ["macro", "micro", "weighted"] + (["samples"] if _task == "multilabel" else [])
        if _task == "binary":
            _averages.append("binary")
        for _average in _averages:
            _names = [_name + "_" + _average]
            if _average == "binary":
                _names += [_name, _name + "_score"]
            if _average == "samples":
                _names.append(_name + "_average_samples")
            _function_with_average = partial(_function, average=_average, zero_division=0)
            _register(
                [_task],
                _names,
                partial(label_metric, metric=_function_with_average, multiclass=_multiclass),
                True,
                ["cutoff", "zero_division"],
            )
    _register(
        [_task],
        ["fbeta", "fbeta_score"],
        partial(
            label_metric,
            metric=partial(skm.fbeta_score, average="macro" if _task != "binary" else "binary", zero_division=0),
            multiclass=_multiclass,
        ),
        True,
        ["beta", "cutoff", "zero_division"],
        ["beta"],
    )

_register(["binary"], ["roc_auc_score"], skm.roc_auc_score, True)
_register(["binary"], ["logloss"], skm.log_loss, False)
_register(["binary"], ["gini"], gini, True)
_register(["binary", "multilabel"], ["pr_auc", "average_precision"], skm.average_precision_score, True)
_register(["binary"], ["brier_loss", "brier_score"], skm.brier_score_loss, False)
_register(["binary", "multiclass"], ["ece"], expected_calibration_error, False, ["n_bins"])
_register(["multilabel"], ["ece"], multilabel_ece, False, ["n_bins"])
_register(["multilabel"], ["brier_loss", "brier_score"], multilabel_brier_loss, False)
_register(["binary"], ["rec@spec", "recall_at_specificity"], recall_at_specificity, True, ["threshold"])
_register(["binary"], ["aucroc@fpr", "partial_auc"], partial_auc, True, ["threshold"])
_register(["multiclass"], ["logloss"], skm.log_loss, False)
for _mode in ("ovr", "ovo"):
    _register(["multiclass"], ["roc_auc_score_" + _mode], partial(skm.roc_auc_score, multi_class=_mode), True)
for _average in ("micro", "macro", "samples"):
    _register(["multilabel"], ["roc_auc_score_" + _average], partial(skm.roc_auc_score, average=_average), True)
# Preserve the work helper's column-wise, NaN-aware semantics under its original alias.
_register(["multilabel"], ["roc_auc_with_nans", "roc_auc_score_average_samples"], roc_auc_with_nans, True)
_register(["multilabel"], ["gini_with_nans"], gini_with_nans, True)
_register(["multilabel"], ["hamming_accuracy"], partial(label_metric, metric=hamming_accuracy), True, ["cutoff"])
for _name in ("label_ranking_average_precision_score", "label_ranking_loss", "coverage_error"):
    _register(["multilabel", "ranking"], [_name], getattr(skm, _name), _name == "label_ranking_average_precision_score")

for _names, _function, _greater in (
    (["r2_score"], skm.r2_score, True),
    (["MAE", "mean_absolute_error"], skm.mean_absolute_error, False),
    (["MSE", "mean_squared_error"], skm.mean_squared_error, False),
    (["RMSE", "rmse"], rmse, False),
    (["MAPE"], mape, False),
    (["median_absolute_error"], skm.median_absolute_error, False),
    (["median_absolute_percentage_error"], median_absolute_percentage_error, False),
    (["mean_squared_log_error", "msle"], skm.mean_squared_log_error, False),
    (["SMAPE", "smape"], smape, False),
    (["1-WAPE", "one_minus_wape"], one_minus_wape, True),
    (["wape", "mae_to_module_mean"], mae_to_module_mean, False),
    (["mae_to_mean"], mae_to_mean, False),
    (["mape_without_zeros"], mape_without_zeros, False),
    (["spearman"], spearman, True),
    (["powerstat"], powerstat, True),
):
    _register(["reg", "multi:reg"], _names, _function, _greater)
_register(["reg", "multi:reg"], ["r2_adj", "adjusted_r2"], adjusted_r2, True, ["n_features"], ["n_features"])

for _names, _function in (
    (["precision@k", "precision_at_k"], precision_at_k),
    (["recall@k", "recall_at_k"], recall_at_k),
    (["ap@k", "average_precision@k", "average_precision_at_k"], average_precision_at_k),
    (["average_precision_in_top_k"], average_precision_in_top_k),
):
    _register(["binary"], _names, _function, True, ["k"])
_register(["multilabel", "ranking"], ["map@k", "mean_average_precision_at_k"], mean_average_precision_at_k, True, ["k"])
_register(
    ["multilabel", "ranking"],
    ["mean_average_precision_in_top_k"],
    partial(mean_average_precision_at_k, within_top_k=True),
    True,
    ["k"],
)
_register(["multilabel", "ranking"], ["ndcg_score", "ndcg@k", "ndcg"], ndcg_at_k, True, ["k"])
_register(["multilabel"], ["mrr"], reciprocal_rank_score, True, ["k"])
_register(["ranking"], ["reciprocal_rank_score"], reciprocal_rank_score, True, ["k"])
_register(["ranking"], ["mrr"], mean_reciprocal_rank, True)
_register(["timeseries"], ["mean_mae"], mean_mae, False, ["start_indices"])
_register(["timeseries"], ["mean_mse"], mean_mse, False, ["start_indices"])
_register(["uplift"], ["qini_auc_score"], qini_auc_score, True, ["negative_effect"])
_register(["uplift"], ["uplift_auc_score"], uplift_auc_score, True)
_register(["uplift"], ["uplift@k", "uplift_at_k"], uplift_at_k, True, ["k", "strategy"])
_register(["uplift"], ["weighted_average_uplift"], weighted_average_uplift, True, ["strategy", "bins"])


def get_metric(name, task="binary", **metric_params):
    """Return a raw metric callable with parameters bound and a greater_is_better attribute."""
    if task not in _METRICS or name not in _METRICS[task]:
        raise ValueError("Unknown metric {!r} for task {!r}.".format(name, task))
    spec = _METRICS[task][name]
    spec.validate(name, metric_params)
    metric = partial(spec.func, **metric_params)
    metric.greater_is_better = spec.greater_is_better
    return metric
