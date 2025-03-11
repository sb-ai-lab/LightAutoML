from sklearn.metrics import roc_auc_score, mean_squared_error

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


class TestTabularAutoMLXGB:
    def test_fit_predict_binary(self, sampled_app_train_test, sampled_app_roles, binary_task):
        # load and prepare data
        train, test = sampled_app_train_test

        # run automl
        automl = TabularAutoML(task=binary_task, general_params={"use_algos": [["xgb"]]})
        oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        target_name = get_target_name(sampled_app_roles)
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

        # checks
        assert oof_score > 0.69
        assert ho_score > 0.69

        check_pickling(automl, ho_score, binary_task, test, target_name)

    def test_fit_predict_multiclass(self, sampled_app_train_test):
        # load and prepare data
        train, test = sampled_app_train_test

        # custom metric
        def _roc_auc_score_ovr(y_true, y_pred) -> float:
            return roc_auc_score(y_true, y_pred, multi_class="ovr")

        task = Task("multiclass", metric=_roc_auc_score_ovr)

        target_name = "NAME_FAMILY_STATUS"
        roles = {"target": target_name}

        # run automl
        automl = TabularAutoML(task=task, general_params={"use_algos": [["xgb"]]})
        oof_predictions = automl.fit_predict(train, roles=roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data, multi_class="ovr")
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data, multi_class="ovr")

        # checks
        assert oof_score > 0.53  # 0.5485819143117492
        assert ho_score > 0.53  # 0.53985643506639

        # check_pickling(automl, ho_score, binary_task, test, target_name) # _roc_auc_score_ovr should be defined in pickle.load

    def test_fit_predict_regression(self, sampled_app_train_test):
        # load and prepare data
        train, test = sampled_app_train_test

        task = Task("reg", metric=mean_squared_error)

        target_name = "AMT_ANNUITY"
        roles = {"target": target_name}

        # run automl
        automl = TabularAutoML(task=task, general_params={"use_algos": [["xgb"]]})
        oof_predictions = automl.fit_predict(train, roles=roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        oof_score = mean_squared_error(train[target_name].values, oof_predictions.data)
        ho_score = mean_squared_error(test[target_name].values, ho_predictions.data)

        # checks
        assert oof_score < 72636914  # 71636914
        assert ho_score < 62123460  # 60123460

        # check_pickling(automl, ho_score, binary_task, test, target_name)
