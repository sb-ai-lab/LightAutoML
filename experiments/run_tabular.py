"""Run tabular automl using ClearML logging."""

from utils import Timer
from utils import install_lightautoml

install_lightautoml()

import argparse
import os
import clearml
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def main(dataset_name: str, cpu_limit: int, memory_limit: int): # noqa D103
    cml_task = clearml.Task.get_task(clearml.config.get_remote_task_id())
    logger = cml_task.get_logger()

    dataset = clearml.Dataset.get(dataset_id=None, dataset_name=dataset_name)
    dataset_local_path = dataset.get_local_copy()

    with open(os.path.join(dataset_local_path, "task_type.txt"), "r") as f:
        task_type = f.readline()
    train = pd.read_csv(os.path.join(dataset_local_path, "train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(dataset_local_path, "test.csv"), index_col=0)

    task = Task(task_type)
    automl = TabularAutoML(task=task, cpu_limit=cpu_limit, memory_limit=memory_limit, timeout=6000)
    cml_task.connect(automl)

    with Timer() as timer_training:
        oof_predictions = automl.fit_predict(train, roles={"target": "class"}, verbose=10)

    with Timer() as timer_predict:
        test_predictions = automl.predict(test)

    if task_type == "binary":
        metric_oof = roc_auc_score(train["class"].values, oof_predictions.data[:, 0])
        metric_ho = roc_auc_score(test["class"].values, test_predictions.data[:, 0])

    elif task_type == "multiclass":
        metric_oof = log_loss(train["class"].map(automl.reader.class_mapping), oof_predictions.data)
        metric_ho = log_loss(test["class"].map(automl.reader.class_mapping), test_predictions.data)

    elif task_type == "regression":
        metric_oof = task.metric_func(train[target].values, oof_predictions.data[:, 0])
        metric_ho = task.metric_func(test[target].values, test_predictions.data[:, 0])

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
