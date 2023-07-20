from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.whitebox_presets import WhiteBoxPreset
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


class TestWhiteBoxPreset:
    def test_fit_predict(self, jobs_train_test, jobs_roles, binary_task):
        # load and prepare data
        train, test = jobs_train_test

        # run automl
        automl = WhiteBoxPreset(binary_task)
        oof_predictions = automl.fit_predict(train.reset_index(drop=True), roles=jobs_roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        target_name = get_target_name(jobs_roles)
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

        # checks
        assert oof_score > 0.78
        assert ho_score > 0.78

        check_pickling(automl, ho_score, binary_task, test, target_name)
