from presets_utils import check_pickling
from presets_utils import get_target_name
from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML


class TestTabularAutoML:
    def test_fit_predict(self, sampled_app_train_test, sampled_app_roles, binary_task):
        # load and prepare data
        train, test = sampled_app_train_test

        # run automl
        automl = TabularAutoML(task=binary_task, linear_l2_params={"default_params": {"max_iter": 200}})
        oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        target_name = get_target_name(sampled_app_roles)
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

        # checks
        assert oof_score > 0.73
        assert ho_score > 0.72

        check_pickling(automl, ho_score, binary_task, test, target_name)
