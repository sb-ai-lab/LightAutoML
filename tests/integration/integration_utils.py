import numpy as np

from lightautoml.dataset.roles import TargetRole
from joblib import load

from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


def load_and_test_automl(filename, task, score, pred, data, target_name):
    automl = load(filename)

    test_pred_joblib = automl.predict(data)

    if task.name == "binary":
        score_new = roc_auc_score(data[target_name].values, test_pred_joblib.data[:, 0])
    elif task.name == "multiclass":
        score_new = log_loss(data[target_name].map(automl.reader.targets_mapping), test_pred_joblib.data)
    elif task.name == "reg":
        score_new = mean_squared_error(data[target_name].values, test_pred_joblib.data[:, 0])

    np.testing.assert_almost_equal(score, score_new, decimal=3)
    np.testing.assert_allclose(pred.data[:, 0], test_pred_joblib.data[:, 0])


def get_target_name(roles):
    for key, value in roles.items():
        if (key == "target") or isinstance(key, TargetRole):
            return value
