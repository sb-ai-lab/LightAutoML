import warnings

import numpy as np

from lightautoml.ml_algo.linear_sklearn import LinearL1CD
from lightautoml.tasks import Task


def test_linear_l1_uses_current_logistic_regression_api():
    algo = LinearL1CD(default_params={"max_iter": 10000})
    algo.task = Task("binary")
    model, _, l1_ratios, _ = algo._infer_params()
    model.set_params(l1_ratio=l1_ratios[0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 1, 1]))

    assert model.get_params()["l1_ratio"] == 1
