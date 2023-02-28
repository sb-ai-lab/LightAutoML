#!/usr/bin/env python
# coding: utf-8

import os
import pickle

import pytest

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.utils import roles_parser
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.validation.np_iterators import FoldsIterator


@pytest.mark.integtest
def test_manual_pipeline(sampled_app_train_test, sampled_app_roles, binary_task):

    train, test = sampled_app_train_test

    pd_dataset = PandasDataset(train, roles_parser(sampled_app_roles), task=binary_task)

    selector_iterator = FoldsIterator(pd_dataset, 1)

    pipe = LGBSimpleFeatures()

    model0 = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "seed": 0,
            "num_threads": 5,
        }
    )

    mbie = ModelBasedImportanceEstimator()
    selector = ImportanceCutoffSelector(pipe, model0, mbie, cutoff=10)

    selector.fit(selector_iterator)

    pipe = LGBSimpleFeatures()

    params_tuner1 = OptunaTuner(n_trials=10, timeout=300)
    model1 = BoostLGBM(default_params={"learning_rate": 0.05, "num_leaves": 128})

    params_tuner2 = OptunaTuner(n_trials=100, timeout=300)
    model2 = BoostLGBM(default_params={"learning_rate": 0.025, "num_leaves": 64})

    total = MLPipeline(
        [(model1, params_tuner1), (model2, params_tuner2)],
        pre_selection=selector,
        features_pipeline=pipe,
        post_selection=None,
    )

    train_valid = FoldsIterator(pd_dataset)

    total.fit_predict(train_valid)

    total.predict(pd_dataset)

    with open("automl.pickle", "wb") as f:
        pickle.dump(total, f)

    with open("automl.pickle", "rb") as f:
        total = pickle.load(f)

    total.predict(pd_dataset)
    os.remove("automl.pickle")

    # assert
