#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tempfile

from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.validation.np_iterators import TimeSeriesIterator


np.random.seed(42)


def test_tabular_with_dates(sampled_app_train_test, binary_task):

    train, test = sampled_app_train_test

    # create time series iterator that is passed as cv_func
    cv_iter = TimeSeriesIterator(train["EMP_DATE"].astype("datetime64[ns]"), n_splits=5, sorted_kfold=False)

    # train dataset may be passed as dict of np.ndarray
    train = {
        "data": train[["AMT_CREDIT", "AMT_ANNUITY"]].values,
        "target": train["TARGET"].values,
    }

    task = binary_task

    automl = TabularAutoML(
        task=task,
        timeout=200,
    )
    oof_pred = automl.fit_predict(train, train_features=["AMT_CREDIT", "AMT_ANNUITY"], cv_iter=cv_iter)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # prediction can be made on file by
        tmp_file = os.path.join(tmpdirname, "temp_test_data.csv")
        test.to_csv(tmp_file, index=False)
        test_pred = automl.predict(tmp_file, batch_size=100, n_jobs=4)

    oof_prediction = oof_pred.data[:, 0]
    not_empty = np.logical_not(np.isnan(oof_prediction))

    oof_score = roc_auc_score(train["target"][not_empty], oof_prediction[not_empty])
    assert oof_score > 0.52

    test_score = roc_auc_score(test["TARGET"].values, test_pred.data[:, 0])
    assert test_score > 0.51
