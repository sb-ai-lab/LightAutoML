from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lightautoml.addons.tabular_interpretation import SSWARM
from lightautoml.addons.tabular_interpretation import sswarm


class SmallModel:
    def __init__(self, n_features=5, task="reg"):
        self.reader = SimpleNamespace(
            used_features=[f"f{i}" for i in range(n_features)],
            _n_classes={"reg": None, "binary": 2, "multiclass": 3}[task],
            task=SimpleNamespace(name=task),
        )
        self.calls = []
        self.fail_on_call = None

    def collect_model_stats(self):
        return {"SmallModel": 1}

    def predict(self, data, features_names=None, n_jobs=1):
        if isinstance(data, pd.DataFrame):
            features_names = list(data.columns)
        assert list(features_names) == self.reader.used_features
        data = np.asarray(data)
        assert not any(value is None for value in data.ravel())
        self.calls.append((len(data), data.dtype, n_jobs))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("Prediction failed")
        values = np.nan_to_num(data.astype(np.float64))
        signal = values[:, 0].copy()
        if values.shape[1] > 1:
            signal += 0.7 * values[:, 1] ** 2
        if values.shape[1] > 2:
            signal += np.sin(values[:, 2])
        if self.reader.task.name == "binary":
            result = (1 / (1 + np.exp(-signal)))[:, None]
        elif self.reader.task.name == "multiclass":
            result = np.exp(np.column_stack([signal, -signal, np.zeros_like(signal)]))
            result /= result.sum(axis=1, keepdims=True)
        else:
            result = signal[:, None]
        return SimpleNamespace(data=result.astype(np.float32))


class FrameModel(SmallModel):
    def predict(self, data):
        assert isinstance(data, pd.DataFrame)
        return super().predict(data)


class RecordingSSWARM(SSWARM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.counts = []

    def exactCalculation(self):
        self.records.append([])
        super().exactCalculation()

    def update(self, A, A_plus, A_minus, v):
        if not self.records[-1]:
            self.counts.append((self.c_plus.copy(), self.c_minus.copy()))
        self.records[-1].append((A, A_plus, A_minus, v.copy()))
        super().update(A, A_plus, A_minus, v)


def make_data(n_rows=11, n_features=5):
    return pd.DataFrame(
        np.random.default_rng(123).normal(size=(n_rows, n_features)).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features)],
    )


def dense_reference(explainer, data):
    """Reproduce the old running stratum means, independently of final counts."""
    v_N = explainer.v(data.to_numpy(), feature_names=list(data.columns))
    result = np.zeros((len(data), explainer.n, explainer.n_outputs))
    for records in explainer.records:
        shape = (len(data), explainer.n, explainer.n, explainer.n_outputs)
        plus, minus = np.zeros(shape), np.zeros(shape)
        c_plus, c_minus = np.zeros(shape), np.zeros(shape)
        for A, A_plus, A_minus, values in records:
            card = len(A)
            for feature in A_plus.intersection(A):
                mean, count = plus[:, feature, card - 1], c_plus[:, feature, card - 1]
                mean[:] = (mean * count + values) / (count + 1)
                count += 1
            for feature in A_minus.difference(A):
                mean, count = minus[:, feature, card], c_minus[:, feature, card]
                mean[:] = (mean * count + values) / (count + 1)
                count += 1
        phi = (plus - minus).sum(axis=2) / explainer.n
        correction = ((v_N - phi.sum(axis=1)) / explainer.n)[:, None, :]
        result += (phi + correction - v_N.mean(axis=0) / explainer.n) / len(explainer.records)
    result = result.transpose(2, 0, 1)
    return result[0] if explainer.n_outputs == 1 else result


def assert_no_working_state(explainer):
    for name in ("_phi", "phi_plus", "phi_minus", "c_plus", "c_minus", "updates"):
        assert not hasattr(explainer, name)


@pytest.mark.parametrize("task", ["reg", "binary", "multiclass"])
@pytest.mark.parametrize("n_repeats", [1, 2])
def test_streaming_matches_dense_stratum_means(task, n_repeats):
    data = make_data()
    original = data.copy(deep=True)
    explainer = RecordingSSWARM(SmallModel(task=task))
    actual = explainer.shap_values(data, T=47, n_repeats=n_repeats, batch_size=44)
    np.testing.assert_allclose(actual, dense_reference(explainer, data), rtol=1e-12, atol=1e-12)
    prediction = explainer.v(data.to_numpy(), feature_names=list(data.columns))
    expected = prediction.mean(axis=0)
    reconstructed = actual.sum(axis=-1) + np.asarray(explainer.expected_value).reshape(-1, 1)
    np.testing.assert_allclose(reconstructed, prediction.T, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(np.atleast_1d(explainer.expected_value), expected)
    pd.testing.assert_frame_equal(data, original)
    assert_no_working_state(explainer)


@pytest.mark.parametrize("task", ["reg", "binary", "multiclass"])
def test_batch_size_and_repeated_calls_do_not_change_sampling(task):
    data = make_data()
    explainer = SSWARM(SmallModel(task=task))
    expected = explainer.shap_values(data, T=47, n_repeats=2, batch_size=len(data))
    for batch_size in (7 * len(data) + 3, 500_000):
        actual = explainer.shap_values(data, T=47, n_repeats=2, batch_size=batch_size)
        np.testing.assert_array_equal(actual, expected)
        assert_no_working_state(explainer)


def test_default_budget_covers_all_warm_up_strata():
    data = make_data(n_rows=2, n_features=60)
    model = SmallModel(60)
    explainer = RecordingSSWARM(model)
    actual = explainer.shap_values(data, n_repeats=1, batch_size=100)
    assert len(explainer.records[0]) == 613  # The previous default stopped at 500.
    plus, minus = explainer.counts[0]
    assert plus.shape == minus.shape == (60, 60)
    assert np.issubdtype(plus.dtype, np.integer)
    assert np.all(plus > 0)
    assert np.all(minus[:, 1:] > 0)
    assert not np.any(minus[:, 0])  # The empty coalition is handled by the SHAP baseline.
    assert sum(rows for rows, _, _ in model.calls) == (613 + 1) * len(data)
    expected = SSWARM(SmallModel(60)).shap_values(data, n_repeats=1, batch_size=500_000)
    np.testing.assert_array_equal(actual, expected)


def test_insufficient_budget_is_rejected_before_prediction():
    model = SmallModel(100)
    with pytest.raises(ValueError, match="T must be at least 1141"):
        SSWARM(model).shap_values(make_data(n_features=100), T=500)
    assert not model.calls


def test_partial_batch_has_no_padding_and_preserves_numeric_dtype():
    data = make_data(n_rows=7, n_features=4)
    model = SmallModel(4)
    SSWARM(model).shap_values(data, T=17, n_repeats=1, batch_size=28)
    assert [rows for rows, _, _ in model.calls] == [7, 28, 28, 28, 28, 7]
    assert all(dtype == np.float32 for _, dtype, _ in model.calls)


@pytest.mark.parametrize("model_class", [SmallModel, FrameModel])
def test_feature_order_unused_columns_and_array_input(model_class):
    data = make_data()
    model = model_class()
    explainer = SSWARM(model)
    kwargs = dict(T=25, n_repeats=1, batch_size=55)
    expected = explainer.shap_values(data, **kwargs)
    data["unused"] = "unused object column"
    reordered = data[list(reversed(data.columns))]
    original = reordered.copy(deep=True)
    np.testing.assert_array_equal(explainer.shap_values(reordered, **kwargs), expected)
    pd.testing.assert_frame_equal(reordered, original)
    assert all(dtype == np.float32 for _, dtype, _ in model.calls)
    numeric = reordered.drop(columns="unused")
    np.testing.assert_array_equal(
        explainer.shap_values(numeric.to_numpy(), feature_names=list(numeric.columns), **kwargs), expected
    )


def test_mixed_used_features_and_missing_values():
    class MixedModel(SmallModel):
        def predict(self, data, features_names=None, n_jobs=1):
            assert data.dtype == object
            data = data.copy()
            assert set(data[:, 1]) <= {"red", "blue"}
            data[:, 1] = np.where(data[:, 1] == "red", 1.0, -1.0)
            return super().predict(data, features_names, n_jobs)

    data = make_data()
    data["f1"] = ["red", "blue"] * 5 + ["red"]
    data.loc[3, "f0"] = np.nan
    original = data.copy(deep=True)
    explainer = SSWARM(MixedModel())
    values = explainer.shap_values(data, T=25, n_repeats=1, batch_size=55)
    assert values.shape == (len(data), 5)
    assert np.isfinite(values).all()
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("task", ["reg", "binary", "multiclass"])
def test_small_feature_sets_use_all_nonempty_coalitions(n_features, task):
    data = make_data(n_features=n_features)
    explainer = RecordingSSWARM(SmallModel(n_features, task))
    actual = explainer.shap_values(data, T=50, n_repeats=1, batch_size=len(data))
    coalitions = [frozenset(record[0]) for record in explainer.records[0]]
    assert len(coalitions) == len(set(coalitions)) == 2 ** n_features - 1
    np.testing.assert_allclose(actual, dense_reference(explainer, data), rtol=1e-12, atol=1e-12)


def test_failure_releases_working_state_and_resets_seed(monkeypatch):
    bars = []

    class Progress:
        def __init__(self, **kwargs):
            self.closed = False
            bars.append(self)

        def update(self, **kwargs):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(sswarm, "tqdm", Progress)
    data = make_data()
    model = SmallModel()
    model.fail_on_call = 3
    explainer = SSWARM(model)
    kwargs = dict(T=25, n_repeats=1, batch_size=len(data))
    with pytest.raises(RuntimeError, match="Prediction failed"):
        explainer.shap_values(data, **kwargs)
    assert_no_working_state(explainer)
    assert bars[-1].closed
    model.fail_on_call = None
    actual = explainer.shap_values(data, **kwargs)
    expected = SSWARM(SmallModel()).shap_values(data, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert all(bar.closed for bar in bars)


def test_accumulator_allocations_do_not_include_observation_stratum_tensors(monkeypatch):
    shapes = []
    zeros = np.zeros

    def record_zeros(shape, *args, **kwargs):
        shapes.append(shape)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(sswarm.np, "zeros", record_zeros)
    explainer = SSWARM(SmallModel(task="multiclass"))
    explainer.shap_values(make_data(), T=25, n_repeats=2, batch_size=55)
    assert shapes
    assert all(not isinstance(shape, tuple) or len(shape) <= 3 for shape in shapes)
    assert_no_working_state(explainer)


@pytest.mark.parametrize("name", ["T", "n_repeats", "batch_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_integer_parameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        SSWARM(SmallModel()).shap_values(make_data(), **{name: value})


@pytest.mark.parametrize(
    "data, names, message",
    [
        (np.ones(5), None, "two-dimensional"),
        (np.ones((1, 5)), None, "at least 2 observations"),
        (np.ones((2, 5)), None, "one name per input column"),
        (np.ones((2, 5)), ["f0"], "one name per input column"),
        (np.ones((2, 5)), ["f0"] * 5, "must be unique"),
        (np.ones((2, 5)), ["f0", "f1", "f2", "f3", "other"], "Missing model features"),
    ],
)
def test_invalid_input(data, names, message):
    with pytest.raises(ValueError, match=message):
        SSWARM(SmallModel()).shap_values(data, feature_names=names)


def test_batch_size_cannot_split_background_observations():
    with pytest.raises(ValueError, match="increase the batch_size"):
        SSWARM(SmallModel()).shap_values(make_data(), batch_size=10)


def test_model_without_features_is_rejected():
    with pytest.raises(ValueError, match="at least one feature"):
        SSWARM(SmallModel(0)).shap_values(make_data())


def test_parallel_prediction_guard_and_forwarding():
    model = SmallModel()
    SSWARM(model).shap_values(make_data(), T=25, n_repeats=1, n_jobs=2)
    assert [jobs for _, _, jobs in model.calls] == [1, 2]
    model.collect_model_stats = lambda: {"TorchNN": 1}
    with pytest.raises(ValueError, match="Set the n_jobs parameter to 1"):
        SSWARM(model).shap_values(make_data(), n_jobs=2)


@pytest.mark.parametrize("task", ["reg", "binary", "multiclass"])
def test_tabular_automl_predictions_and_dense_reference(task):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    data = make_data(n_rows=160, n_features=8)
    signal = data["f0"] + 2 * data["f1"]
    if task == "binary":
        data["target"] = (signal > 0).astype(int)
    elif task == "multiclass":
        data["target"] = np.digitize(signal, [-1, 1])
    else:
        data["target"] = signal
    model = TabularAutoML(
        task=Task(task),
        timeout=30,
        cpu_limit=1,
        gpu_ids="",
        debug=True,
        general_params={"use_algos": [["lgb"]]},
        reader_params={"cv": 2, "advanced_roles": False, "random_state": 123},
        lgb_params={"default_params": {"num_trees": 3, "early_stopping_rounds": 2, "min_data_in_leaf": 5, "seed": 123}},
    )
    model.fit_predict(data, roles={"target": "target"}, verbose=0)
    observations = data.iloc[:16]  # Including an unused target must not affect prediction.
    explainer = RecordingSSWARM(model)
    actual = explainer.shap_values(observations, T=80, n_repeats=2, batch_size=128)
    np.testing.assert_allclose(actual, dense_reference(explainer, observations), rtol=1e-12, atol=1e-12)
    expected_shape = (16, 8) if task == "reg" else (model.reader._n_classes, 16, 8)
    assert actual.shape == expected_shape
    prediction = explainer.v(observations.to_numpy(), feature_names=list(observations.columns))
    reconstructed = actual.sum(axis=-1) + np.asarray(explainer.expected_value).reshape(-1, 1)
    np.testing.assert_allclose(reconstructed, prediction.T, rtol=1e-6, atol=1e-7)
    assert_no_working_state(explainer)
