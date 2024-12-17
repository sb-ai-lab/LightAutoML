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


def map_to_corect_order_of_classes(values, targets_order):  # noqa D103
    target_mapping = {n: x for (x, n) in enumerate(targets_order)}
    mapped = list(map(target_mapping.get, values))

    return mapped


def main(dataset_name: str, cpu_limit: int, memory_limit: int, save_model: bool):  # noqa D103
    cml_task = clearml.Task.get_task(clearml.config.get_remote_task_id())
    logger = cml_task.get_logger()

    dataset = clearml.Dataset.get(dataset_id=None, dataset_name=dataset_name)
    dataset_local_path = dataset.get_local_copy()

    with open(os.path.join(dataset_local_path, "task_type.txt"), "r") as f:
        task_type = f.readline()
    train = pd.read_csv(os.path.join(dataset_local_path, "train.csv"))
    test = pd.read_csv(os.path.join(dataset_local_path, "test.csv"))

    if task_type == "multilabel":
        target_name = [x for x in test.columns if x.startswith("target")]
    else:
        target_name = test.columns[-1]

    if task_type in ["binary", "multiclass", "multilabel"]:
        assert (
            train[target_name].nunique() == test[target_name].nunique()
        ), "train and test has different unique values."

        is_train_unique_ok = train[target_name].nunique() > 1
        is_test_unique_ok = test[target_name].nunique() > 1

        if isinstance(is_train_unique_ok, bool):
            assert is_train_unique_ok, "Only one class present in train target."
        else:
            (is_train_unique_ok).all(), "Only one class present in train target."

        if isinstance(is_test_unique_ok, bool):
            assert is_test_unique_ok, "Only one class present in test target."
        else:
            (is_test_unique_ok).all(), "Only one class present in test target."

    assert train[target_name].isnull().values.any() is np.False_, "train has nans in target."
    assert test[target_name].isnull().values.any() is np.False_, "test has nans in target."

    task = Task(task_type)

    # =================================== automl config:
    automl = TabularAutoML(
        debug=True,
        task=task,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        timeout=15 * 60,
        # general_params={
        # "use_algos": [["mlp"]]
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

    kwargs = {}
    if save_model:
        kwargs["path_to_save"] = "model"

    with Timer() as timer_training:
        oof_predictions = automl.fit_predict(train, roles={"target": target_name}, verbose=10, **kwargs)

    # add and upload local file artifact
    cml_task.upload_artifact(
        name="model.joblib",
        artifact_object=os.path.join(
            "model.joblib",
        ),
    )

    with Timer() as timer_predict:
        test_predictions = automl.predict(test)

    if task_type == "binary":
        print(f"OOF: {oof_predictions.data[:, 0].unique()}")
        metric_oof = roc_auc_score(train[target_name].values, oof_predictions.data[:, 0])
        metric_ho = roc_auc_score(test[target_name].values, test_predictions.data[:, 0])

    elif task_type == "multiclass":
        not_nan = np.any(~np.isnan(oof_predictions.data), axis=1)
        try:
            metric_oof = log_loss(train[target_name].values[not_nan], oof_predictions.data[not_nan, :])
            metric_ho = log_loss(test[target_name], test_predictions.data)
        except:
            if np.unique(train[target_name].values[not_nan]).shape != np.unique(oof_predictions.data[not_nan, :]).shape:
                raise ValueError(f"Vectors have different number of classes: {np.unique(train[target_name].values[not_nan])} and {np.unique(oof_predictions.data[not_nan, :])}")
            # Some datasets can have dtype=float of target,
            # so we must map this target for correct log_loss calculating (if we didn't calсulate it in the try block)
            # and this mapping must be in the correct order so we extract automl.targets_order and map values
            y_true = map_to_corect_order_of_classes(
                values=train[target_name].values[not_nan], targets_order=automl.targets_order
            )
            metric_oof = log_loss(y_true, oof_predictions.data[not_nan, :])

            y_true = map_to_corect_order_of_classes(values=test[target_name], targets_order=automl.targets_order)

            metric_ho = log_loss(y_true, test_predictions.data)

    elif task_type == "reg":
        metric_oof = task.metric_func(train[target_name].values, oof_predictions.data[:, 0])
        metric_ho = task.metric_func(test[target_name].values, test_predictions.data[:, 0])

    elif task_type == "multilabel":
        metric_oof = task.metric_func(train[target_name].values, oof_predictions.data)
        metric_ho = task.metric_func(test[target_name].values, test_predictions.data)
    else:
        raise ValueError("Bad task type.")

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
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset, cpu_limit=args.cpu_limit, memory_limit=args.memory_limit, save_model=args.save_model
    )
