"""Check existing native loss adapters against their mathematical targets."""

import pickle

from types import SimpleNamespace

import numpy as np
import pytest
import xgboost

from sklearn.metrics import log_loss
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import mean_squared_error

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_xgb import BoostXGB
from lightautoml.tasks import Task
from lightautoml.tasks.losses.base import fw_rmsle
from lightautoml.tasks.losses.lgb import LGBLoss
from lightautoml.tasks.losses.lgb_custom import lgb_f1_loss_multiclass
from lightautoml.tasks.losses.xgb import XGBLoss
from lightautoml.validation.base import HoldoutIterator


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("target_scale", [1.0, 1e6])
def test_xgboost_rmsle_learns_the_optimal_constant(weighted, target_scale):
    target = np.tile(np.array([0.0, 3.0, 15.0, 63.0]), 30) * target_scale
    weights = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 30) if weighted else None
    task = Task("reg", loss="rmsle", metric="rmsle")
    dataset = NumpyDataset(
        np.zeros((120, 1)),
        ["constant"],
        NumericRole(),
        task,
        target=target,
        **({"weights": weights} if weighted else {}),
    )
    model = BoostXGB(
        default_params={
            "n_estimators": 500,
            "early_stopping_rounds": 30,
            "nthread": 1,
            "learning_rate": 0.1,
            "reg_lambda": 0,
            "verbose_eval": False,
        }
    )
    valid = dataset[80:]

    result = model.fit_predict(HoldoutIterator(dataset[:80], valid)).data[:, 0]

    optimum = np.expm1(np.average(np.log1p(target), weights=weights))
    np.testing.assert_allclose(result, optimum, rtol=0.01)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(valid).data[:, 0], result)
    expected_score = np.sqrt(np.average((np.log1p(target[80:]) - np.log1p(result)) ** 2, weights=valid.weights))
    assert model.models[0].best_score == pytest.approx(expected_score, rel=1e-5)


@pytest.mark.parametrize("weighted", [False, True])
def test_rmsle_can_early_stop_on_original_scale_metric(weighted):
    target = np.tile(np.array([0.0, 3.0, 15.0, 63.0]), 30)
    weights = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 30)
    dataset = NumpyDataset(
        np.zeros((120, 1)),
        ["constant"],
        NumericRole(),
        Task("reg", loss="rmsle", metric="mse"),
        target=target,
        **({"weights": weights} if weighted else {}),
    )
    model = BoostXGB(default_params={"n_estimators": 10, "nthread": 1, "verbose_eval": False})
    valid = dataset[80:]

    predicted = model.fit_predict(HoldoutIterator(dataset[:80], valid)).data[:, 0]

    expected = mean_squared_error(valid.target, predicted, sample_weight=valid.weights)
    assert model.models[0].best_score == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("metric", ["quantile", "mse"])
@pytest.mark.parametrize("q", [0.2, 0.8])
def test_xgboost_quantile_honors_q_and_metric_parameters(metric, q):
    target = np.tile(np.arange(20, dtype=np.float32), 10)
    task = Task("reg", loss="quantile", loss_params={"q": q}, metric=metric)
    dataset = NumpyDataset(np.zeros((200, 1)), ["constant"], NumericRole(), task, target=target)
    model = BoostXGB(
        default_params={
            "n_estimators": 100,
            "early_stopping_rounds": 10,
            "nthread": 1,
            "learning_rate": 0.3,
            "verbose_eval": False,
        }
    )

    result = model.fit_predict(HoldoutIterator(dataset[:120], dataset[120:])).data[:, 0]

    if metric == "quantile":
        assert abs(result[0] - np.quantile(target, q)) < 1
        expected = mean_pinball_loss(target[120:], result, alpha=q)
        assert model.models[0].best_score == pytest.approx(expected, abs=1e-5)
    assert np.isfinite(result).all()
    assert task.losses["xgb"].fobj_params == {"quantile_alpha": q}


def test_xgboost_quantile_callback_uses_observation_weights():
    task = Task("reg", loss="quantile", loss_params={"q": 0.8}, metric="quantile")
    target = np.array([1.0, 5.0, 10.0])
    predicted = np.array([3.0, 3.0, 3.0])
    weights = np.array([1.0, 2.0, 4.0])
    matrix = xgboost.DMatrix(np.zeros((3, 1)), label=target, weight=weights)

    name, actual = task.losses["xgb"].feval(predicted, matrix)

    assert name == "Opt_metric"
    assert actual == pytest.approx(mean_pinball_loss(target, predicted, alpha=0.8, sample_weight=weights))
    assert task.losses["xgb"].metric_params == {}


def test_xgboost_huber_parameters_are_native_and_do_not_mutate_input():
    parameters = {"a": 0.3}
    task = Task("reg", loss="huber", loss_params=parameters, metric="huber")
    dataset = NumpyDataset(
        np.arange(120, dtype=np.float32).reshape(60, 2),
        ["a", "b"],
        NumericRole(),
        task,
        target=np.linspace(0, 2, 60),
    )
    model = BoostXGB(default_params={"n_estimators": 4, "nthread": 1, "verbose_eval": False})

    result = model.fit_predict(HoldoutIterator(dataset[:40], dataset[40:]))

    assert task.losses["xgb"].fobj_params == {"huber_slope": 0.3}
    assert parameters == {"a": 0.3}
    assert np.isfinite(result.data).all()


@pytest.mark.parametrize("key,factory", [("lgb", LGBLoss), ("xgb", XGBLoss)])
def test_task_accepts_explicit_supported_loss_adapters(key, factory):
    loss = factory("mse")
    assert Task("reg", loss={key: loss}).losses == {key: loss}


@pytest.mark.parametrize("keys", [[], ["unknown"], ["lgb", "unknown"]])
def test_task_rejects_invalid_loss_adapter_keys(keys):
    with pytest.raises(AssertionError, match="Invalid loss key"):
        Task("reg", loss={key: LGBLoss("mse") for key in keys})


@pytest.mark.parametrize("flat", [False, True])
def test_multiclass_f1_derivatives_keep_input_shape_and_weights(flat):
    predictions = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2], [0.2, 0.3, 0.1]])
    if flat:
        predictions = predictions.reshape(-1, order="F")
    target = np.array([0, 1, 2])
    weights = np.array([0.0, 2.0, 3.0])
    plain = SimpleNamespace(get_label=lambda: target, get_weight=lambda: None)
    weighted = SimpleNamespace(get_label=lambda: target, get_weight=lambda: weights)

    plain_grad, plain_hess = lgb_f1_loss_multiclass(predictions, plain)
    grad, hess = lgb_f1_loss_multiclass(predictions, weighted)

    assert grad.shape == hess.shape == predictions.shape
    np.testing.assert_allclose(grad.reshape(3, 3, order="F"), plain_grad.reshape(3, 3, order="F") * weights[:, None])
    np.testing.assert_allclose(hess.reshape(3, 3, order="F"), plain_hess.reshape(3, 3, order="F") * weights[:, None])


@pytest.mark.parametrize("kind", ["lgb", "xgb"])
def test_transformed_regression_metric_restores_labels_and_predictions(kind):
    loss = LGBLoss("rmsle") if kind == "lgb" else XGBLoss("reg:squarederror", fw_func=fw_rmsle, bw_func=np.expm1)
    callback = loss.metric_wrapper(mean_squared_error, False)
    target = np.array([0.0, 3.0, 15.0])
    predicted = np.array([1.0, 4.0, 10.0])
    weights = np.array([1.0, 2.0, 4.0])
    data = SimpleNamespace(get_label=lambda: np.log1p(target), get_weight=lambda: weights)

    actual = callback(np.log1p(predicted), data)[1]

    assert actual == pytest.approx(mean_squared_error(target, predicted, sample_weight=weights))


@pytest.mark.parametrize("target", [[0, 1, 0], [0, 0, 0]])
@pytest.mark.parametrize("weights", [None, [1.0, 2.0, 3.0]])
def test_multiclass_crossentropy_keeps_absent_classes(target, weights):
    predicted = [[0.6, 0.2, 0.2], [0.2, 0.7, 0.1], [0.5, 0.1, 0.4]]
    expected = log_loss(target, predicted, sample_weight=weights, labels=[0, 1, 2])

    assert Task("multiclass").metric_func(target, predicted, sample_weight=weights) == pytest.approx(expected)


@pytest.mark.parametrize(
    "target,predicted",
    [
        (["a", "b", "c"], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        ([2, 4, 6], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        ([0, 1, 1], [0.2, 0.7, 0.8]),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    ],
)
def test_crossentropy_preserves_existing_sklearn_input_formats(target, predicted):
    expected = log_loss(target, predicted)
    assert Task("multiclass").metric_func(target, predicted) == pytest.approx(expected)
