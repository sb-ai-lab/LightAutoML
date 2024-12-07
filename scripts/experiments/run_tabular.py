"""Run tabular automl using ClearML logging."""

from utils import Timer
from utils import install_lightautoml


install_lightautoml()

import argparse
import os

import clearml
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

RANDOM_STATE = 1234


def main(dataset_name: str, cpu_limit: int, memory_limit: int):  # noqa D103
    cml_task = clearml.Task.get_task(clearml.config.get_remote_task_id())
    logger = cml_task.get_logger()

    dataset = clearml.Dataset.get(dataset_id=None, dataset_name=dataset_name)
    dataset_local_path = dataset.get_local_copy()

    with open(os.path.join(dataset_local_path, "task_type.txt"), "r") as f:
        task_type = f.readline()
    train = pd.read_csv(os.path.join(dataset_local_path, "train.csv"))
    test = pd.read_csv(os.path.join(dataset_local_path, "test.csv"))

    task = Task(task_type)

    # =================================== automl config:
    automl = TabularAutoML(
        debug=True,
        task=task,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        timeout=10 * 60 * 60,
        # general_params={
        #     "use_algos": [["mlp"]]
        # },  # ['nn', 'mlp', 'dense', 'denselight', 'resnet', 'snn', 'node', 'autoint', 'fttransformer'] or custom torch model
        # nn_params={"n_epochs": 10, "bs": 512, "num_workers": 0, "path_to_save": None, "freeze_defaults": True},
        # nn_pipeline_params={"use_qnt": True, "use_te": False},
        reader_params={
            #     # 'n_jobs': N_THREADS,
            #     "cv": 5,
            "random_state": RANDOM_STATE,
        },
    )
    # ===================================

    cml_task.connect(automl)

    target_name = test.columns[-1]

    with Timer() as timer_training:
        oof_predictions = automl.fit_predict(train, roles={"target": target_name}, verbose=10)

    with Timer() as timer_predict:
        test_predictions = automl.predict(test)

    if task_type == "binary":
        metric_oof = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        metric_ho = roc_auc_score(test[target_name].values, test_predictions.data[:, 0])

    elif task_type == "multiclass":
        not_nan = np.any(~np.isnan(oof_predictions.data), axis=1)
        metric_oof = log_loss(train[target_name].values[not_nan], oof_predictions.data[not_nan, :])
        metric_ho = log_loss(test[target_name], test_predictions.data)

    elif task_type == "reg":
        metric_oof = task.metric_func(train[target_name].values, oof_predictions.data[:, 0])
        metric_ho = task.metric_func(test[target_name].values, test_predictions.data[:, 0])

    print(f"Score for out-of-fold predictions: {metric_oof}")
    print(f"Score for hold-out: {metric_ho}")
    print(f"Train duration: {timer_training.duration}")
    print(f"Predict duration: {timer_predict.duration}")

    logger.report_single_value("Metric OOF", metric_oof)
    logger.report_single_value("Metric HO", metric_ho)

    logger.report_single_value("Train duration", timer_training.duration)
    logger.report_single_value("Predict duration", timer_predict.duration)

    logger.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="dataset name or id", default="sampled_app_train")
    parser.add_argument("--cpu_limit", type=int, help="", default=8)
    parser.add_argument("--memory_limit", type=int, help="", default=16)
    args = parser.parse_args()

    main(dataset_name=args.dataset, cpu_limit=args.cpu_limit, memory_limit=args.memory_limit)
