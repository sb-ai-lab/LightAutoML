#!/usr/bin/env python
# coding: utf-8

import tempfile
from os.path import join as pjoin
from sklearn.metrics import roc_auc_score

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.reader.base import PandasToPandasReader

from integration_utils import get_target_name, load_and_test_automl


def test_manual_pipeline(sampled_app_train_test, sampled_app_roles, binary_task):

    train, test = sampled_app_train_test
    target_name = get_target_name(sampled_app_roles)

    reader = PandasToPandasReader(binary_task, cv=5, random_state=1)

    # Create feature selector that uses importances to cutoff features set
    selector = ImportanceCutoffSelector(
        LGBSimpleFeatures(),
        BoostLGBM(
            default_params={
                "learning_rate": 0.05,
                "num_leaves": 64,
                "seed": 42,
                "num_threads": 5,
            }
        ),
        ModelBasedImportanceEstimator(),
        cutoff=10,
    )

    # Or you can use iteartive feature selection algorithm:
    # selector = NpIterativeFeatureSelector(
    #     LGBSimpleFeatures(),
    #     BoostLGBM(
    #         default_params={
    #             "learning_rate": 0.05,
    #             "num_leaves": 64,
    #             "seed": 42,
    #             "num_threads": 5,
    #         }
    #     ),
    #     NpPermutationImportanceEstimator(),
    #     feature_group_size=1,
    #     max_features_cnt_in_result=15
    # )

    # Build pipeline for 1st level
    pipeline_lvl1 = MLPipeline(
        [
            (
                BoostLGBM(
                    default_params={
                        "learning_rate": 0.05,
                        "num_leaves": 128,
                        "seed": 1,
                        "num_threads": 5,
                    }
                ),
                OptunaTuner(n_trials=100, timeout=300),
            ),
            BoostLGBM(
                default_params={
                    "learning_rate": 0.025,
                    "num_leaves": 64,
                    "seed": 2,
                    "num_threads": 5,
                }
            ),
        ],
        pre_selection=selector,
        features_pipeline=LGBSimpleFeatures(),
        post_selection=None,
    )

    # Build pipeline for 2nd level

    pipeline_lvl2 = MLPipeline(
        [
            BoostLGBM(
                default_params={
                    "learning_rate": 0.05,
                    "num_leaves": 64,
                    "max_bin": 1024,
                    "seed": 3,
                    "num_threads": 5,
                },
                freeze_defaults=True,
            )
        ],
        pre_selection=None,
        features_pipeline=LGBSimpleFeatures(),
        post_selection=None,
    )

    # Create AutoML with 2 level stacking
    automl = AutoML(
        reader,
        [
            [pipeline_lvl1],
            [pipeline_lvl2],
        ],
        skip_conn=False,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        path_to_save = pjoin(tmpdirname, "model.joblib")
        # Start AutoML training
        oof_predictions = automl.fit_predict(
            train,
            roles=sampled_app_roles,
            path_to_save=path_to_save,
        )

        # predict for test data
        ho_predictions = automl.predict(test)

        oof_score = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        ho_score = roc_auc_score(test[target_name].values, ho_predictions.data[:, 0])

        assert oof_score > 0.67
        assert ho_score > 0.67

        load_and_test_automl(
            filename=path_to_save,
            task=binary_task,
            score=ho_score,
            pred=ho_predictions,
            data=test,
            target_name=target_name,
        )
