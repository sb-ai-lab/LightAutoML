#!/usr/bin/env python
# coding: utf-8
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.features.base import TabularDataFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task
from lightautoml.transformers.base import ChangeRoles
from lightautoml.transformers.base import ColumnsSelector
from lightautoml.transformers.base import SequentialTransformer
from lightautoml.transformers.base import UnionTransformer
from lightautoml.transformers.categorical import LabelEncoder
from lightautoml.transformers.composite import GroupByTransformer
from lightautoml.transformers.numeric import FillnaMedian
from lightautoml.validation.np_iterators import TimeSeriesIterator


################################
# Features:
# - group_by transformer
################################

N_FOLDS = 3  # folds cnt for AutoML
RANDOM_STATE = 42  # fixed random state for various reasons
N_THREADS = 4  # threads cnt for lgbm and linear models


class GroupByPipeline(FeaturesPipeline, TabularDataFeatures):
    def __init__(
        self, feats_imp=None, top_category: int = 3, top_numeric: int = 3, **kwargs
    ):
        """Helper class to create pipeline with group_by transformer."""

        super().__init__(feats_imp=feats_imp)

        self.top_group_by_categorical = top_category
        self.top_group_by_numerical = top_numeric

    def create_pipeline(self, train):
        return self.get_group_by(train)


def test_groupby_transformer():
    np.random.seed(RANDOM_STATE)

    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.WARNING
    )

    data = pd.read_csv("../example_data/test_data_files/sampled_app_train.csv")

    data["BIRTH_DATE"] = (
        np.datetime64("2018-01-01")
        + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    ).astype(str)
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01")
        + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)

    data["report_dt"] = np.datetime64("2018-01-01")

    data["constant"] = 1
    data["allnan"] = np.nan

    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

    train, test = train_test_split(data, test_size=2000, random_state=42)

    # create time series iterator that is passed as cv_func
    cv_iter = TimeSeriesIterator(
        train["EMP_DATE"].astype(np.datetime64), n_splits=5, sorted_kfold=False
    )

    task = Task(
        "binary",
    )

    roles = {
        "target": "TARGET",
    }

    reader = PandasToPandasReader(
        task, cv=N_FOLDS, random_state=RANDOM_STATE, advanced_roles=False
    )

    model = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 128,
            "seed": 1,
            "num_threads": N_THREADS,
        }
    )

    pipe = GroupByPipeline(None, top_category=4, top_numeric=4)

    pipeline = MLPipeline(
        [
            (model),
        ],
        features_pipeline=pipe,
    )

    automl = AutoML(
        reader,
        [
            [pipeline],
        ],
        skip_conn=False,
        verbose=1,
    )

    oof_pred = automl.fit_predict(
        train,
        train_features=["AMT_CREDIT", "AMT_ANNUITY"],
        cv_iter=cv_iter,
        roles=roles,
    )

    test_pred = automl.predict(test)

    logging.debug("Check scores...")
    oof_prediction = oof_pred.data[:, 0]
    not_empty = np.logical_not(np.isnan(oof_prediction))
    logging.debug(
        "OOF score: {}".format(
            roc_auc_score(train["TARGET"][not_empty], oof_prediction[not_empty])
        )
    )
    logging.debug(
        "TEST score: {}".format(
            roc_auc_score(test["TARGET"].values, test_pred.data[:, 0])
        )
    )
