import logging.config
import uuid

import pandas as pd
from sklearn.model_selection import train_test_split

from examples_utils import get_dataset_attrs
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.tasks import Task

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


def main(dataset_name: str, seed: int):
    cv = 5

    # Algos and layers to be used during automl:
    # For example:
    # 1. use_algos = [["lgb"]]
    # 2. use_algos = [["linear_l2"]]
    # 3. use_algos = [["lgb", "linear_l2"], ["lgb"]]
    use_algos = [["lgb", "linear_l2"], ["lgb"]]

    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    with log_exec_timer("LAMA") as train_timer:
        data = pd.read_csv(path, dtype=dtype)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

        task = Task(task_type)

        num_threads = 8
        automl = TabularAutoML(
            task=task,
            cpu_limit=num_threads,
            timeout=3600 * 3,
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False},
            lgb_params={"default_params": {"num_threads": num_threads}},
            # linear_l2_params={"default_params": {"cs": [1e-5]}},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"Score for out-of-fold predictions: {metric_value}")

    with log_exec_timer() as predict_timer:
        te_pred = automl.predict(test_data)
        te_pred.target = test_data[roles['target']]

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

    logger.info(f"Score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    result = {
        "seed": seed,
        "dataset": dataset_name,
        "used_algo": str(use_algos),
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration
    }

    print(f"EXP-RESULT: {result}")

    return result


def multirun(dataset_name: str):
    seeds = [ 1, 5, 42, 100, 777]
    results = [main(dataset_name, seed) for seed in seeds]

    df = pd.DataFrame(results)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    df.to_csv(f"lama_results_{dataset_name}_{uuid.uuid4()}.csv")


if __name__ == "__main__":
    # One can run:
    # 1. main(dataset_name="used_cars_dataset", seed=42)
    # 2. multirun(dataset_name="used_cars_dataset")
    main(dataset_name="used_cars_dataset", seed=42)
    # multirun(dataset_name="used_cars_dataset_1x")
