import numpy as np
import pytest

import lightautoml.utils.logging  # noqa: F401

from lightautoml.tasks import Task


@pytest.mark.parametrize("backend", ["lgb", "xgb"])
@pytest.mark.parametrize(
    "name,params",
    [
        ("precision_score", {"cutoff": 0.8}),
        ("precision@k", {"k": 2}),
        ("ece", {"n_bins": 2}),
        ("pr_auc", {}),
    ],
)
def test_callback_receives_metric_parameters_weights_and_direction(backend, name, params):
    task = Task("binary", metric=name, metric_params=params)
    target, pred, weights = np.array([0, 1, 1, 0]), np.array([0.1, 0.75, 0.8, 0.9]), np.array([1, 2, 3, 4])

    class Dataset:
        def get_label(self):
            return target

        def get_weight(self):
            return weights

    loss = task.losses[backend]
    result = loss.feval(pred, Dataset())
    assert result[1] == pytest.approx(task.metric_func(target, pred, sample_weight=weights))
    assert loss.feval.greater_is_better == task.greater_is_better
    assert loss.metric_params == {}
    if backend == "lgb":
        assert loss.metric_name == "None"
        assert result[2] == task.greater_is_better


@pytest.mark.parametrize("backend", ["lgb", "xgb", "cb"])
@pytest.mark.parametrize(
    "task_name,metric,metric_params",
    [
        ("binary", "pr_auc", {}),
        ("binary", "fbeta_score", {"beta": 0.5, "cutoff": 0.4}),
        ("reg", "SMAPE", {}),
        ("multiclass", "precision_macro", {}),
    ],
)
def test_training_with_new_metrics(backend, task_name, metric, metric_params, tmp_path):
    rng = np.random.RandomState(12)
    features = rng.normal(size=(120, 4))
    if task_name == "binary":
        target = (features[:, 0] + features[:, 1] > 0).astype(int)
    elif task_name == "multiclass":
        target = np.tile([0, 1, 2], 40)
    else:
        target = 5 + features[:, 0]
    weights = rng.uniform(1, 2, 120)
    task = Task(task_name, metric=metric, metric_params=metric_params)
    loss = task.losses[backend]
    train_x, valid_x, train_y, valid_y = features[:80], features[80:], target[:80], target[80:]
    if backend == "lgb":
        import lightgbm as lgb

        train = lgb.Dataset(train_x, label=train_y, weight=weights[:80])
        valid = lgb.Dataset(valid_x, label=valid_y, weight=weights[80:], reference=train)
        params = {
            "objective": loss.fobj_name,
            "metric": loss.metric_name,
            "num_threads": 2,
            "verbosity": -1,
            "seed": 42,
            "min_data_in_leaf": 5,
        }
        if task_name == "multiclass":
            params["num_class"] = 3
        history = {}
        model = lgb.train(
            params,
            train,
            num_boost_round=5,
            valid_sets=[valid],
            valid_names=["valid"],
            feval=loss.feval,
            callbacks=[lgb.record_evaluation(history)],
        )
        pred = model.predict(valid_x, num_threads=2)
        observed = history["valid"]["Opt metric"][-1]
        assert list(history["valid"]) == ["Opt metric"]
    elif backend == "xgb":
        import xgboost as xgb

        train = xgb.DMatrix(train_x, label=train_y, weight=weights[:80])
        valid = xgb.DMatrix(valid_x, label=valid_y, weight=weights[80:])
        params = {"objective": loss.fobj_name, "nthread": 2, "seed": 42}
        if task_name == "multiclass":
            params["num_class"] = 3
        history = {}
        model = xgb.train(
            params,
            train,
            num_boost_round=5,
            evals=[(valid, "valid")],
            custom_metric=loss.feval,
            evals_result=history,
            verbose_eval=False,
        )
        pred = model.predict(valid)
        observed = history["valid"]["Opt_metric"][-1]
    else:
        from catboost import CatBoost
        from catboost import Pool

        train = Pool(train_x, label=train_y, weight=weights[:80])
        valid = Pool(valid_x, label=valid_y, weight=weights[80:])
        assert loss.metric_name == loss.fobj_name
        model = CatBoost(
            {
                "loss_function": loss.fobj_name,
                "eval_metric": loss.metric_name,
                "iterations": 5,
                "thread_count": 2,
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
            }
        )
        model.fit(train, eval_set=valid)
        pred = model.predict(valid, prediction_type="RawFormulaVal" if task_name == "reg" else "Probability")
        if task_name == "binary":
            pred = pred[:, 1]
        observed = task.metric_func(valid_y, pred, sample_weight=weights[80:])
    expected = task.metric_func(valid_y, pred, sample_weight=weights[80:])
    assert np.isfinite(expected)
    assert observed == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "task_name,metric,objective",
    [("multi:reg", "SMAPE", "MultiRMSE"), ("multilabel", "hamming_accuracy", "MultiCrossEntropy")],
)
def test_catboost_multioutput_objective_is_preserved(task_name, metric, objective):
    task = Task(task_name, metric=metric)
    assert task.losses["cb"].fobj_name == objective
    assert task.losses["cb"].metric_name == objective


def test_explicit_optimization_direction_overrides_registry():
    task = Task("binary", metric="ece", greater_is_better=True)
    assert task.greater_is_better
    assert task.losses["lgb"].feval.greater_is_better
    assert task.losses["xgb"].feval.greater_is_better
