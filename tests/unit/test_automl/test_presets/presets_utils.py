import os
import pickle
from pytest import approx
import tempfile

from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from lightautoml.dataset.roles import TargetRole


def check_pickling(automl, ho_score, task, test_data, target_name):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "automl.pickle")
        with open(filename, "wb") as f:
            pickle.dump(automl, f)

        with open(filename, "rb") as f:
            automl = pickle.load(f)

        test_pred = automl.predict(test_data)

        if task.name == "binary":
            ho_score_new = roc_auc_score(test_data[target_name].values, test_pred.data[:, 0])
        elif task.name == "multiclass":
            ho_score_new = log_loss(test_data[target_name].map(automl.reader.target_mapping), test_pred.data)
        elif task.name == "reg":
            ho_score_new = mean_squared_error(test_data[target_name].values, test_pred.data[:, 0])

        assert ho_score == approx(ho_score_new, rel=1e-3)


def get_target_name(roles):
    for key, value in roles.items():
        if (key == "target") or isinstance(key, TargetRole):
            return value
