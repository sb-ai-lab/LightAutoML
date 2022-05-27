import logging
import logging.config
import os
import shutil
from copy import deepcopy, copy
from pprint import pprint
from typing import Callable

from dataset_utils import datasets

from lama_experiments import calculate_automl as lama_automl
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.utils.tmp_utils import LOG_DATA_DIR, log_config
from spark_experiments import calculate_automl as spark_automl, calculate_lgbadv_boostlgb

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='./lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def calculate_quality(calc_automl: Callable, delete_dir: bool = True):

    # dataset_name = "used_cars_dataset_0125x"
    # dataset_name = "used_cars_dataset_head2x60k"
    # dataset_name = "lama_test_dataset"
    dataset_name = "ipums_97"

    config = copy(datasets()[dataset_name])
    # config["use_algos"] = [["lgb", "linear_l2"], ["lgb"]]
    # config["use_algos"] = [["lgb", "linear_l2"]]
    # config["use_algos"] = [["linear_l2"], ["lgb"]]
    config["use_algos"] = [["lgb"]]
    # config["use_algos"] = [["linear_l2"]]

    cv = 5
    seeds = [42]
    results = []
    for seed in seeds:
        cfg = deepcopy(config)
        cfg['seed'] = seed
        cfg['cv'] = cv

        cfg['checkpoint_path'] = '/opt/checkpoints/tmp_chck'
        log_config("general", cfg)

        res = calc_automl(**cfg)
        results.append(res)
        logger.info(f"Result for seed {seed}: {res}")

    mvals = [f"{r['metric_value']:_.2f}" for r in results]
    print("OOf on train metric")
    pprint(mvals)

    test_mvals = [f"{r['test_metric_value']:_.2f}" for r in results]
    print("Test metric")
    pprint(test_mvals)


if __name__ == "__main__":
    calculate_quality(lama_automl)
    # calculate_quality(spark_automl, delete_dir=False)
    # calculate_quality(calculate_lgbadv_boostlgb, delete_dir=False)
