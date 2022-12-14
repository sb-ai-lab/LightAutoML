from utils import install_lightautoml
from utils import Timer
install_lightautoml()

import argparse
import clearml
import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def main(dataset_name, cpu_limit, memory_limit):
    cml_task = clearml.Task.get_task(clearml.config.get_remote_task_id())
    logger = cml_task.get_logger()

    dataset = clearml.Dataset.get(dataset_id=None, dataset_name=dataset_name)
    dataset_local_path = dataset.get_local_copy()

    with open(os.path.join(dataset_local_path, 'task_type.txt'), 'r') as f:
        task_type = f.readline()
    train = pd.read_csv(os.path.join(dataset_local_path, 'train.csv'))
    test = pd.read_csv(os.path.join(dataset_local_path, 'test.csv'))

    automl = TabularAutoML(task=Task(task_type), cpu_limit=cpu_limit, memory_limit=memory_limit)
    cml_task.connect(automl)

    with Timer() as timer_training:
        oof_predictions = automl.fit_predict(train, roles={"target": "TARGET"}, verbose=10) # TODO reuse target 
    
    with Timer() as timer_predict:
        te_pred = automl.predict(test)

    if task_type == "binary":
        metric_oof = roc_auc_score(train['TARGET'].values, oof_predictions.data[:, 0])
        metric_ho = roc_auc_score(test['TARGET'].values, te_pred.data[:, 0])
    elif task_type == "regression": # TODO
        metric_oof = roc_auc_score(train['TARGET'].values, oof_predictions.data[:, 0])
        metric_ho = roc_auc_score(test['TARGET'].values, te_pred.data[:, 0])
    elif task_type == "multiclass": # TODO
        metric_oof = roc_auc_score(train['TARGET'].values, oof_predictions.data[:, 0])
        metric_ho = roc_auc_score(test['TARGET'].values, te_pred.data[:, 0])

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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, help='dataset name or id', default='sampled_app_train')
    parser.add_argument('--cpu_limit', type=str, help='', default=8)
    parser.add_argument('--memory_limit', type=str, help='', default=16)
    args = parser.parse_args()
    
    main(dataset_name=args.dataset, cpu_limit=args.cpu_limit, memory_limit=args.memory_limit)
