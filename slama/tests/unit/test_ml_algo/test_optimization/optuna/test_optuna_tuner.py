from unittest import mock

import pytest

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.base import Distribution
from lightautoml.ml_algo.tuning.base import SearchSpace
from lightautoml.ml_algo.tuning.optuna import OptunaTuner


# from lightautoml.dataset.np_pd_dataset import PandasDataset
# from lightautoml.dataset.utils import roles_parser
# from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
# from lightautoml.validation.np_iterators import FoldsIterator


# @pytest.mark.parametrize(
#     "sampled_app_train_test",
#     [
#         (1000),
#     ],
#     indirect=["sampled_app_train_test"],
# )
# def test_params_values_ranges(
#     sampled_app_train_test,
#     sampled_app_roles,
#     binary_task,
# ):

#     train, _ = sampled_app_train_test

#     features_pipeline = LGBSimpleFeatures()
#     iterator = FoldsIterator(
#         PandasDataset(
#             data=train,
#             roles=roles_parser(sampled_app_roles),
#             task=binary_task,
#         )
#     )

#     iterator = iterator.apply_feature_pipeline(features_pipeline)

#     model = BoostLGBM(
#         default_params={"num_trees": 1, "random_state": 42},
#         freeze_defaults=True,
#         optimization_search_space={
#             "feature_fraction": SearchSpace(Distribution.UNIFORM, low=0.5, high=1.0),
#             "min_sum_hessian_in_leaf": SearchSpace(Distribution.CHOICE, choices=[0.5, 0.8]),
#         },
#     )

#     params_tuner = OptunaTuner(n_trials=10, timeout=300)
#     params_tuner.fit(
#         ml_algo=model,
#         train_valid_iterator=iterator,
#     )

#     # check that the hyperparameters values are in the difined search space
#     for trial in params_tuner.study.get_trials():
#         assert (trial.params["feature_fraction"] >= 0) and (trial.params["feature_fraction"] <= 1)
#         assert trial.params["min_sum_hessian_in_leaf"] in [0.5, 0.8]

#     # check time, n_trials

#     # check best params
#     assert (params_tuner.best_params["feature_fraction"] == 0.7993292420985183) and (
#         params_tuner.best_params["min_sum_hessian_in_leaf"] == 0.5
#     )


def test_invalid_distributions():
    iterator_mock = mock.MagicMock()

    model = BoostLGBM(
        default_params={"num_trees": 1, "random_state": 42},
        freeze_defaults=True,
        optimization_search_space={
            "feature_fraction": SearchSpace(Distribution.UNIFORM, low=0.5, high=1.0),
            "min_sum_hessian_in_leaf": SearchSpace(
                Distribution.NORMAL, something=0
            ),  # distribution is not supported by Optuna
        },
    )

    params_tuner = OptunaTuner(n_trials=10, timeout=300)

    with pytest.raises(ValueError):
        params_tuner.fit(
            ml_algo=model,
            train_valid_iterator=iterator_mock,
        )
