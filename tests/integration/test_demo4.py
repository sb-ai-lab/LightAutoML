import numpy as np

from sklearn.metrics import roc_auc_score

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task

np.random.seed(42)


def test_different_task_params(sampled_app_train_test):

    train_data, test_data = sampled_app_train_test

    for task_params, target in zip(
        [
            {"name": "binary"},
            {"name": "binary", "metric": roc_auc_score},
            {"name": "reg", "loss": "mse", "metric": "r2"},
            {"name": "reg", "loss": "rmsle", "metric": "rmsle"},
            {
                "name": "reg",
                "loss": "quantile",
                "loss_params": {"q": 0.9},
                "metric": "quantile",
                "metric_params": {"q": 0.9},
            },
        ],
        ["TARGET", "TARGET", "AMT_CREDIT", "AMT_CREDIT", "AMT_CREDIT"],
    ):

        task = Task(**task_params)

        reader = PandasToPandasReader(task, cv=5, random_state=1)

        # pipeline 1 level parts
        pipe = LGBSimpleFeatures()

        model2 = BoostLGBM(
            default_params={
                "learning_rate": 0.025,
                "num_leaves": 64,
                "seed": 2,
                "num_threads": 5,
            }
        )

        pipeline_lvl1 = MLPipeline(
            [model2],
            pre_selection=None,  # selector,
            features_pipeline=pipe,
            post_selection=None,
        )

        automl = AutoML(
            reader,
            [
                [pipeline_lvl1],
            ],
            skip_conn=False,
            # debug=True,
        )

        oof_pred = automl.fit_predict(train_data, roles={"target": target}, verbose=1)
    # assert for last oof score
    assert task.metric_func(train_data[target].values, oof_pred.data[:, 0]) < 10 ** 5
    assert task.metric_func(test_data[target].values, automl.predict(test_data).data[:, 0]) < 10 ** 5
