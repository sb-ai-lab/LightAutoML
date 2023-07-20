import numpy as np

from sklearn.metrics import mean_squared_error

from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


class TestTabularNLPAutoML:
    def test_fit_predict(self, avito1k_train_test, avito1k_roles, regression_task):
        # load and prepare data
        train, test = avito1k_train_test

        # run automl
        automl = TabularNLPAutoML(task=regression_task, timeout=600)
        oof_pred = automl.fit_predict(train, roles=avito1k_roles, verbose=10)
        test_pred = automl.predict(test)
        not_nan = np.any(~np.isnan(oof_pred.data), axis=1)

        target_name = get_target_name(avito1k_roles)
        oof_score = mean_squared_error(train[target_name].values[not_nan], oof_pred.data[not_nan][:, 0])
        ho_score = mean_squared_error(test[target_name].values, test_pred.data[:, 0])

        # checks
        assert oof_score < 0.7
        assert ho_score < 0.7

        check_pickling(automl, ho_score, regression_task, test, target_name)
