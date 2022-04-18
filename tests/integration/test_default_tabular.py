import pytest

from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML


@pytest.mark.integtest
def test_default_tabular(sampled_app_train_test, sampled_app_roles, binary_task):
    # load and prepare data
    train, test = sampled_app_train_test

    # run automl
    automl = TabularAutoML(task=binary_task)
    oof_predictions = automl.fit_predict(train, roles=sampled_app_roles, verbose=2)
    te_pred = automl.predict(test)

    # calculate scores
    print(f"Score for out-of-fold predictions: {roc_auc_score(train['TARGET'].values, oof_predictions.data[:, 0])}")
    print(f"Score for hold-out: {roc_auc_score(test['TARGET'].values, te_pred.data[:, 0])}")
    # add check
