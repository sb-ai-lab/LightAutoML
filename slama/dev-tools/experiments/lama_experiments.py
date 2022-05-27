# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
import logging.config
import os
from typing import Dict, Any, Optional

import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split
from dataset_utils import datasets

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer
from lightautoml.tasks import Task
from lightautoml.utils.tmp_utils import log_data, LAMA_LIBRARY

logger = logging.getLogger(__name__)


def calculate_automl(path: str,
                     task_type: str,
                     metric_name: str,
                     target_col: str = 'target',
                     seed: int = 42,
                     cv: int = 5,
                     use_algos = ("lgb", "linear_l2"),
                     roles: Optional[Dict] = None,
                     dtype: Optional[Dict] = None,
                     **_) -> Dict[str, Any]:
    os.environ[LAMA_LIBRARY] = "lama"

    with log_exec_timer("LAMA") as train_timer:
        # to assure that LAMA correctly interprets these columns as categorical
        roles = roles if roles else {}
        dtype = dtype if dtype else {}

        data = pd.read_csv(path,  dtype=dtype)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

        task = Task(task_type)

        automl = TabularAutoML(
            task=task,
            timeout=3600 * 3,
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"{metric_name} score for out-of-fold predictions: {metric_value}")

    with log_exec_timer() as predict_timer:
        te_pred = automl.predict(test_data)

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

    logger.info(f"mse score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    return {
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration
    }


# if __name__ == "__main__":
#     logging.config.dictConfig(logging_config(level=logging.INFO, log_filename="/tmp/lama.log"))
#     logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
#     logger = logging.getLogger(__name__)
#
#     # Read values from config file
#     with open("/scripts/config.yaml", "r") as stream:
#         config_data = yaml.safe_load(stream)
#
#     ds_cfg = datasets()[config_data['dataset']]
#     del config_data['dataset']
#     ds_cfg.update(config_data)
#
#     result = calculate_automl(**ds_cfg)
#     print(f"EXP-RESULT: {result}")

if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename="/tmp/lama.log"))
    logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)

    # Read values from config file
    with open(config_path, "r") as stream:
        config_data = yaml.safe_load(stream)

    func_name = config_data['func']

    if 'dataset' in config_data:
        ds_cfg = datasets()[config_data['dataset']]
    else:
        ds_cfg = dict()

    ds_cfg.update(config_data)

    if func_name == "calculate_automl":
        func = calculate_automl
    else:
        raise ValueError(f"Incorrect func name: {func_name}.")

    result = func(**ds_cfg)
    print(f"EXP-RESULT: {result}")
