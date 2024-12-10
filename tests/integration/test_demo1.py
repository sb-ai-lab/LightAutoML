#!/usr/bin/env python
# coding: utf-8

import os
import pickle

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


def test_cutoff_selector_in_pipeline(sampled_app_train_test, binary_task):

    train_data, test_data = sampled_app_train_test

    task = binary_task

    reader = PandasToPandasReader(task, cv=5, random_state=1)

    # selector parts
    model0 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 42,
            "num_threads": 5,
        }
    )
    pipe0 = LGBSimpleFeatures()
    mbie = ModelBasedImportanceEstimator()
    selector = ImportanceCutoffSelector(pipe0, model0, mbie, cutoff=10)

    # pipeline 1 level parts
    pipe = LGBSimpleFeatures()

    params_tuner1 = OptunaTuner(n_trials=100, timeout=300)
    model1 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 128,
            "seed": 1,
            "num_threads": 5,
        }
    )
    model2 = BoostLGBM(
        default_params={
            "learning_rate": 0.025,
            "num_leaves": 64,
            "seed": 2,
            "num_threads": 5,
        }
    )

    pipeline_lvl1 = MLPipeline(
        [(model1, params_tuner1), model2],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )

    # pipeline 2 level parts
    pipe1 = LGBSimpleFeatures()

    model = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_bin": 1024,
            "seed": 3,
            "num_threads": 5,
        },
        freeze_defaults=True,
    )

    pipeline_lvl2 = MLPipeline([model], pre_selection=None, features_pipeline=pipe1, post_selection=None)

    automl = AutoML(
        reader,
        [
            [pipeline_lvl1],
            [pipeline_lvl2],
        ],
        skip_conn=False,
        debug=True,
    )

    automl.fit_predict(train_data, roles={"target": "TARGET"}, verbose=5)

    # just checking if methods can be called
    selector.get_features_score()
    automl.levels[-1][0].ml_algos[0].get_features_score()
    automl.levels[0][0].ml_algos[0].get_features_score()
    automl.levels[0][0].ml_algos[1].get_features_score()

    test_pred = automl.predict(test_data)

    with open("automl.pickle", "wb") as f:
        pickle.dump(automl, f)

    with open("automl.pickle", "rb") as f:
        automl = pickle.load(f)

    test_pred = automl.predict(test_data)
    test_score = roc_auc_score(test_data["TARGET"].values, test_pred.data[:, 0])

    assert test_score > 0.65

    os.remove("automl.pickle")
