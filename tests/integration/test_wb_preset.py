import pytest

from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.whitebox_presets import WhiteBoxPreset


@pytest.mark.integtest
def tests_wb_preset(jobs_train_test, binary_task):
    # load and prepare data
    train, test = jobs_train_test

    # run automl
    automl = WhiteBoxPreset(binary_task)
    _ = automl.fit_predict(train.reset_index(drop=True), roles={"target": "target"}, verbose=2)
    test_prediction = automl.predict(test).data[:, 0]

    # calculate scores
    print(f"ROCAUC score: {roc_auc_score(test['target'].values, test_prediction)}")
