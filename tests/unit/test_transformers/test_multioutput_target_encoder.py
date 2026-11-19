"""Regression coverage for independent multioutput target encoding."""

import numpy as np
import pytest

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.tasks import Task
from lightautoml.transformers.categorical import MultioutputTargetEncoder


@pytest.mark.parametrize(
    "task_name,target",
    [
        ("multi:reg", [[2.0, -2.0], [4.0, -4.0], [6.0, -6.0], [8.0, -8.0]]),
        ("multilabel", [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        ("multilabel", [[0.0, 0.0]] * 4),
    ],
)
def test_encoding_preserves_independent_target_means(task_name, target):
    target = np.asarray(target, dtype=np.float32)
    folds = np.array([0, 1, 0, 1])
    dataset = NumpyDataset(
        np.ones((4, 1)), ["category"], CategoryRole(np.int32), task=Task(task_name), target=target, folds=folds
    )
    encoder = MultioutputTargetEncoder(alphas=(1.0,))
    with np.errstate(all="raise"):
        encoded = encoder.fit_transform(dataset)
    expected = np.stack([target[folds != fold].mean(axis=0) for fold in folds])
    np.testing.assert_allclose(encoded.data, expected)
    np.testing.assert_allclose(encoder.transform(dataset).data, np.tile(target.mean(axis=0), (4, 1)))
    assert encoded.features == ["multioof_0__category", "multioof_1__category"]


@pytest.mark.parametrize("task_name", ["multi:reg", "multilabel"])
def test_smoothing_selection_uses_all_targets(task_name):
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    candidates = np.array([[[0.1, 0.2], [0.1, 0.8]], [[0.9, 0.8], [0.9, 0.2]]])
    score = (
        MultioutputTargetEncoder.reg_score_func
        if task_name == "multi:reg"
        else MultioutputTargetEncoder.class_score_func
    )
    assert score(candidates, target) == 1
    assert score(candidates[:, ::-1], target[:, ::-1]) == 1


@pytest.mark.parametrize("task_name", ["multi:reg", "multilabel"])
def test_single_target_and_unseen_category(task_name):
    target = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    dataset = NumpyDataset(
        np.ones((4, 1)),
        ["category"],
        CategoryRole(np.int32),
        task=Task(task_name),
        target=target,
        folds=np.array([0, 1, 0, 1]),
    )
    encoder = MultioutputTargetEncoder(alphas=(1.0,))
    encoded = encoder.fit_transform(dataset)
    np.testing.assert_allclose(encoded.data[:, 0], [1.0, 0.0, 1.0, 0.0])
    unseen = NumpyDataset(np.zeros((2, 1)), ["category"], CategoryRole(np.int32), task=dataset.task)
    np.testing.assert_allclose(encoder.transform(unseen).data, [[0.5], [0.5]])
