from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import ModelBasedImportanceEstimator
from lightautoml.pipelines.selection.permutation_importance_based import NpIterativeFeatureSelector
from lightautoml.pipelines.selection.permutation_importance_based import NpPermutationImportanceEstimator
from lightautoml.tasks import Task
from lightautoml.validation.base import CustomIterator
from lightautoml.validation.base import HoldoutIterator
from lightautoml.validation.np_iterators import FoldsIterator


def dataset(n=240, forced=""):
    rng = np.random.RandomState(82)
    x = rng.normal(size=(n, 6)).astype("float32")
    names = list("abcdef")
    y = x[:, 0] + x[:, 1] + rng.normal(size=n) * 0.2
    return NumpyDataset(
        x,
        names,
        {name: NumericRole(force_input=name in forced) for name in names},
        task=Task("reg"),
        target=y,
        folds=np.arange(n) % 3,
        weights=rng.uniform(0.1, 2, n),
    )


def lgb():
    return BoostLGBM(
        default_params={"num_trees": 12, "early_stopping_rounds": 3, "num_leaves": 6, "num_threads": 1},
        freeze_defaults=True,
    )


@pytest.mark.parametrize("limit,group", [(0, 1), (-1, 1), (1.5, 1), (True, 1), (2, 0), (2, -1), (2, None)])
def test_invalid_counts(limit, group):
    with pytest.raises(ValueError, match="positive integer"):
        NpIterativeFeatureSelector(None, feature_group_size=group, max_features_cnt_in_result=limit)


@pytest.mark.parametrize(
    "limit,group,forced,expected", [(3, 5, "", 3), (2, 1, "ef", 2), (1, 5, "f", 1), (None, 5, "", 6)]
)
def test_budget_and_forced_baseline(monkeypatch, limit, group, forced, expected):
    ds = dataset(forced=forced)
    it = HoldoutIterator(ds, ds)
    ranking = pd.Series(range(6, 0, -1), index=ds.features)
    calls = []

    def fit_candidate(algo, tuner, subset):
        calls.append(subset.features)
        return SimpleNamespace(score=lambda _: len(subset.features)), None

    monkeypatch.setattr(
        "lightautoml.pipelines.selection.permutation_importance_based.tune_and_fit_predict", fit_candidate
    )
    selector = NpIterativeFeatureSelector(
        None,
        lgb(),
        imp_estimator=SimpleNamespace(get_features_score=lambda: ranking),
        feature_group_size=group,
        max_features_cnt_in_result=limit,
    )
    selector._in_features = ds.features
    selector.forced_features = list(forced)
    selector.perform_selection(it)
    selected = selector.select(ds).features
    assert len(selected) == expected
    assert set(forced) <= set(selected)
    assert not set(selected).intersection(selector.dropped_features)
    if limit is not None:
        assert all(len(cols) <= limit for cols in calls)


def test_forced_baseline_can_reject_first_group(monkeypatch):
    ds = dataset(forced="f")
    it = HoldoutIterator(ds, ds)
    calls = []

    def constant_score(algo, tuner, subset):
        calls.append(subset.features)
        return SimpleNamespace(score=lambda _: 1), None

    monkeypatch.setattr(
        "lightautoml.pipelines.selection.permutation_importance_based.tune_and_fit_predict", constant_score
    )
    selector = NpIterativeFeatureSelector(None, lgb(), ModelBasedImportanceEstimator(), max_features_cnt_in_result=2)
    selector.fit(it)
    assert calls[0] == ["f"]
    assert len(calls) > 1
    assert selector.selected_features == ["f"]
    assert set(selector.dropped_features) == set("abcde")
    ds = dataset(forced="ef")
    it = HoldoutIterator(ds, ds)
    with pytest.raises(ValueError, match="Forced features"):
        NpIterativeFeatureSelector(None, max_features_cnt_in_result=1).fit(it)


def test_no_transform_and_prefitted_permutation():
    ds = dataset()
    it = FoldsIterator(ds)
    algo = lgb()
    oof = algo.fit_predict(it)
    gain = ImportanceCutoffSelector(None, algo, ModelBasedImportanceEstimator(), fit_on_holdout=False)
    gain.fit(it)
    assert set(gain.get_features_score().index) == set(ds.features)
    selector = ImportanceCutoffSelector(None, algo, NpPermutationImportanceEstimator(), fit_on_holdout=False)
    with pytest.raises(ValueError, match="OOF predictions"):
        selector.fit(it)
    selector.fit(it, validation_predictions=oof)
    assert set(selector.get_features_score().index) == set(ds.features)
    holdout = HoldoutIterator(ds, ds[:80])
    selector = ImportanceCutoffSelector(None, algo, NpPermutationImportanceEstimator())
    selector.fit(holdout)
    assert selector.selected_features


def test_permutation_matches_weighted_oof_with_overlapping_and_uncovered_rows():
    ds = dataset()
    rng = np.random.RandomState(43)
    indices = np.arange(ds.shape[0])
    splits = [(indices[100:], indices[:100]), (indices[:70], indices[70:180])]
    it = CustomIterator(ds, splits)
    algo = lgb()
    oof = algo.fit_predict(it)
    original = ds.data.copy()
    estimator = NpPermutationImportanceEstimator()
    estimator.fit(it, algo, oof)

    total = np.zeros_like(oof.data)
    count = np.zeros(len(total), dtype=int)
    for model, (rows, _, valid) in zip(algo.models, it):
        perm = rng.permutation(len(rows))
        valid["a"] = valid[:, "a"].data.copy()[perm]
        total[rows, 0] += algo.predict_single_fold(model, valid).ravel()
        count[rows] += 1
    covered = count > 0
    pred = total[covered, 0] / count[covered]
    loss = np.average((ds.target[covered] - pred) ** 2, weights=ds.weights[covered])
    base_loss = np.average((ds.target[covered] - oof.data[covered, 0]) ** 2, weights=ds.weights[covered])
    assert estimator.get_features_score()["a"] == pytest.approx(loss - base_loss)
    np.testing.assert_array_equal(ds.data, original)


def test_reader_preserves_user_folds_for_selection():
    from lightautoml.reader.base import PandasToPandasReader

    supplied = np.repeat([2, 0, 1], 12)
    frame = pd.DataFrame({"feature": np.arange(36), "y": np.arange(36) % 2, "part": supplied})
    reader = PandasToPandasReader(Task("binary"), cv=3, advanced_roles=False)
    ds = reader.fit_read(frame, roles={"target": "y", "folds": "part"})
    for n, (rows, _, _) in enumerate(FoldsIterator(ds)):
        np.testing.assert_array_equal(rows, np.flatnonzero(supplied == n))


def test_time_selection_holdout_respects_dates_in_unsorted_input():
    from lightautoml.validation.np_iterators import TimeSeriesIterator

    dates = np.array([4, 1, 5, 2, 0, 3])
    folds = TimeSeriesIterator.split_by_parts(dates, 3)
    np.testing.assert_array_equal(folds[np.argsort(dates)], [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize("bad", [[0, 2, 0, 2], [0, 1, np.nan, 1]])
def test_invalid_fold_ids_fail_before_feature_selection(bad):
    from lightautoml.reader.base import PandasToPandasReader

    reader = PandasToPandasReader(Task("binary"), cv=2, advanced_roles=False)
    data = pd.DataFrame({"x": [3, 1, 5, 2], "outcome": [0, 1, 0, 1], "fold_id": bad})
    with pytest.raises(ValueError, match="Explicit folds"):
        reader.fit_read(data, roles={"target": "outcome", "folds": "fold_id"})
