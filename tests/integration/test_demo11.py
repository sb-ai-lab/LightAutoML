#!/usr/bin/env python
# coding: utf-8

import shutil

import numpy as np

from sklearn.metrics import mean_squared_error

from lightautoml.automl.presets.text_presets import TabularNLPAutoML


np.random.seed(42)


def test_tabularnlp(avito1k_train_test, avito1k_roles, regression_task):
    train, test = avito1k_train_test

    roles = avito1k_roles

    task = regression_task

    automl = TabularNLPAutoML(task=task, timeout=600)
    oof_pred = automl.fit_predict(train, roles=roles)
    test_pred = automl.predict(test)
    not_nan = np.any(~np.isnan(oof_pred.data), axis=1)
    target = roles["target"]

    oof_score = mean_squared_error(train[target].values[not_nan], oof_pred.data[not_nan][:, 0])

    assert oof_score < 0.2

    test_score = mean_squared_error(test[target].values, test_pred.data[:, 0])
    assert test_score < 0.2

    shutil.rmtree("./models", ignore_errors=True)
