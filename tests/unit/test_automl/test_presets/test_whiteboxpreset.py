import pytest

from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.whitebox_presets import WhiteBoxPreset
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


# AutoWoE uses deprecated scikit-learn logistic parameters: https://github.com/sb-ai-lab/AutoMLWhitebox/issues/29
@pytest.mark.filterwarnings(
    r"ignore:The default value for l1_ratios will change from None:FutureWarning:sklearn\.linear_model\._logistic"
)
@pytest.mark.filterwarnings(
    r"ignore:'penalty' was deprecated in version 1\.8:FutureWarning:sklearn\.linear_model\._logistic"
)
@pytest.mark.filterwarnings(
    r"ignore:The fitted attributes of LogisticRegressionCV:FutureWarning:sklearn\.linear_model\._logistic"
)
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
        assert oof_score > 0.75
        assert ho_score > 0.75

        check_pickling(automl, ho_score, binary_task, test, target_name)
