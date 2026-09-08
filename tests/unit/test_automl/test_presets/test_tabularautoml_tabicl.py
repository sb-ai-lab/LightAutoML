import warnings

import pytest
import torch
from sklearn.metrics import roc_auc_score

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


def require_gpu():
    """Пропускает тест, если GPU недоступна для TabICL."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.cuda")
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            pytest.skip("GPU недоступна для TabICL")


class TestTabularAutoML_TabICL:
    def test_fit_predict_binary(self, sampled_app_train_test, sampled_app_roles, binary_task):
        require_gpu()
        # load and prepare data
        train, test = sampled_app_train_test

        # run automl
        automl = TabularAutoML(task=binary_task, general_params={"use_algos": [["tabicl"]]})
        oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        target_name = get_target_name(sampled_app_roles)
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

        # checks
        assert oof_score > 0.731
        assert ho_score > 0.722

        check_pickling(automl, ho_score, binary_task, test, target_name)

    def test_fit_predict_multiclass(self, sampled_app_train_test):
        require_gpu()
        # load and prepare data
        train, test = sampled_app_train_test

        # custom metric
        def _roc_auc_score_ovr(y_true, y_pred) -> float:
            return roc_auc_score(y_true, y_pred, multi_class="ovr")

        task = Task("multiclass", metric=_roc_auc_score_ovr)

        target_name = "NAME_FAMILY_STATUS"
        roles = {"target": target_name}

        # run automl
        automl = TabularAutoML(task=task, general_params={"use_algos": [["tabicl"]]})
        oof_predictions = automl.fit_predict(train, roles=roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data, multi_class="ovr")
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data, multi_class="ovr")

        # checks
        assert oof_score > 0.528
        assert ho_score > 0.525
