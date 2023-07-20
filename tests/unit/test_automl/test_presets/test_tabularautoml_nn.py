from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


class TestTabularAutoML:
    """Neural network test based on out-of-fold and test scores."""

    def test_fit_predict(self, sampled_app_train_test, sampled_app_roles, binary_task):
        """Test function."""
        # load and prepare data
        train, test = sampled_app_train_test

        # run automl
        automl = TabularAutoML(
            debug=True,
            task=binary_task,
            general_params={"use_algos": [["mlp"]]},
            nn_params={"n_epochs": 10, "bs": 128, "num_workers": 0, "path_to_save": None, "freeze_defaults": True},
        )
        oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)
        ho_predictions = automl.predict(test)

        # calculate scores
        target_name = get_target_name(sampled_app_roles)
        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

        # checks
        assert oof_score > 0.61
        assert ho_score > 0.61

        check_pickling(automl, ho_score, binary_task, test, target_name)
