# -*- encoding: utf-8 -*-

"""Simple example for conditional parameters with OptunaTuner."""

import copy

import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


# load and prepare data
data = pd.read_csv("./data/sampled_app_train.csv")
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=42)


def sample(optimization_search_space, trial, suggested_params):
    trial_values = copy.copy(suggested_params)
    trial_values["feature_fraction"] = trial.suggest_uniform("feature_fraction", low=0.5, high=1.0)

    if trial_values["feature_fraction"] > 0.7:
        trial_values["min_sum_hessian_in_leaf"] = trial.suggest_uniform("min_sum_hessian_in_leaf", low=0.5, high=1)
    else:
        trial_values["min_sum_hessian_in_leaf"] = trial.suggest_uniform("min_sum_hessian_in_leaf", low=0, high=0.5)

    return trial_values


# run automl with custom search spaces
automl = TabularAutoML(
    task=Task("binary"),
    lgb_params={"optimization_search_space": sample},
)
oof_predictions = automl.fit_predict(train_data, roles={"target": "TARGET", "drop": ["SK_ID_CURR"]})
te_pred = automl.predict(test_data)

# calculate scores
print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
