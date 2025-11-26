import sys
import warnings

from sklearn.metrics import roc_auc_score
import pytest

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from tests.unit.test_automl.test_presets.presets_utils import check_pickling
from tests.unit.test_automl.test_presets.presets_utils import get_target_name


class TestTabM:
    """Neural network test based on out-of-fold and test scores."""

    general_params = {"use_algos": [["tabm"]]}

    nn_params = {
        "n_epochs": 10,
        "bs": 128,
        "num_workers": 0,
        "path_to_save": None,
        "freeze_defaults": True,
        "cont_embedder": "piecewise",
        "share_training_batches": False,
    }

    def test_fit_predict(self, sampled_app_train_test, sampled_app_roles, binary_task):
        """Test function."""
        # load and prepare data
        train, test = sampled_app_train_test

        with warnings.catch_warnings():  # TODO: remove filter
            warnings.simplefilter("ignore")

            # run automl
            automl = TabularAutoML(
                debug=True,
                task=binary_task,
                general_params=self.general_params,
                nn_params=self.nn_params,
            )

            if sys.version_info < (3, 9):
                with pytest.raises(RuntimeError) as excinfo:
                    oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)
                assert (
                    "TabM requires Python 3.9+ and the 'tabm' package. Please upgrade Python or install tabm: pip install tabm"
                    in str(excinfo.value)
                )
                return
            else:
                oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)

            ho_predictions = automl.predict(test)

            # calculate scores
            target_name = get_target_name(sampled_app_roles)
            oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
            ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

            # checks
            assert oof_score > 0.64
            assert ho_score > 0.63

            check_pickling(automl, ho_score, binary_task, test, target_name)

    def test_fit_predict_share_training_batches(self, sampled_app_train_test, sampled_app_roles, binary_task):
        """Test function."""
        # load and prepare data
        train, test = sampled_app_train_test

        # run automl

        with warnings.catch_warnings():  # TODO: remove filter
            warnings.simplefilter("ignore")
            automl = TabularAutoML(
                debug=True,
                task=binary_task,
                general_params=self.general_params,
                nn_params={**self.nn_params, **{"share_training_batches": True}},
            )

            if sys.version_info < (3, 9):
                with pytest.raises(RuntimeError) as excinfo:
                    oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)
                assert (
                    "TabM requires Python 3.9+ and the 'tabm' package. Please upgrade Python or install tabm: pip install tabm"
                    in str(excinfo.value)
                )
                return
            else:
                oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=10)

            ho_predictions = automl.predict(test)

            # calculate scores
            target_name = get_target_name(sampled_app_roles)
            oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
            ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

            # checks
            assert oof_score > 0.62
            assert ho_score > 0.62

            check_pickling(automl, ho_score, binary_task, test, target_name)
