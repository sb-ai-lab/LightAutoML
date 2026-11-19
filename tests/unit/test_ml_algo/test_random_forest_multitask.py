"""Random forest multioutput probability handling."""

import numpy as np
import pytest

from sklearn.metrics import roc_auc_score

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.random_forest import RandomForestSklearn
from lightautoml.tasks import Task


@pytest.mark.parametrize("n_targets", [1, 3])
@pytest.mark.parametrize("constant", [0, 1])
def test_multilabel_constant_fold_and_single_output(n_targets, constant):
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    target = np.full((12, n_targets), constant)
    if n_targets > 1:
        target[:, 1] = np.arange(12) % 2
        target[:, 2] = 1 - constant
    task = Task("multilabel", metric=roc_auc_score, greater_is_better=True)
    train = NumpyDataset(x, ["a", "b"], NumericRole(), task=task, target=target)
    valid_target = np.tile((np.arange(12) % 2)[:, None], (1, n_targets))
    valid = NumpyDataset(x, ["a", "b"], NumericRole(), task=task, target=valid_target)
    algo = RandomForestSklearn(default_params={"n_estimators": 3, "max_features": 1.0, "n_jobs": 1})
    algo.task = task
    model, prediction = algo.fit_predict_single_fold(train, valid)
    assert prediction.shape == (12, n_targets)
    np.testing.assert_allclose(prediction[:, 0], constant)
    np.testing.assert_allclose(prediction, algo.predict_single_fold(model, valid))
    assert np.isfinite(prediction).all()
