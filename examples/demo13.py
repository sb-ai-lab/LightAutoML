import logging
import time

import pandas as pd

from sklearn.model_selection import train_test_split

from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task


RANDOM_STATE = 42  # fixed random state for various reasons
USED_COLS = ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL", "DAYS_BIRTH"]


def test_groupby_transformer():
    logging.basicConfig(format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.DEBUG)

    logging.debug("Load data...")
    data = pd.read_csv("./data/sampled_app_train.csv")
    data = data[USED_COLS]
    logging.debug("Data loaded")

    logging.debug("Split data...")
    train_data, test_data = train_test_split(data, test_size=2000, stratify=data["TARGET"], random_state=RANDOM_STATE)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    logging.debug(
        "Data splitted. Parts sizes: train_data = {}, test_data = {}".format(train_data.shape, test_data.shape)
    )

    logging.debug("Create task...")
    task = Task("binary")
    logging.debug("Task created")

    logging.debug("Create reader...")
    reader = PandasToPandasReader(task, cv=5, random_state=RANDOM_STATE)
    logging.debug("Reader created")

    logging.debug("Create MLPipeline pipeline...")
    pipe1 = LGBAdvancedPipeline(use_group_by=True)
    model = BoostLGBM(
        default_params={
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_bin": 1024,
            "seed": RANDOM_STATE,
            "num_threads": 5,
        },
        freeze_defaults=True,
    )
    pipeline = MLPipeline([model], pre_selection=None, features_pipeline=pipe1, post_selection=None)
    logging.debug("MLPipeline pipeline created")

    logging.debug("Create AutoML pipeline...")
    automl = AutoML(
        reader,
        [[pipeline]],
        skip_conn=False,
    )
    logging.debug("AutoML pipeline created")

    logging.debug("Start AutoML pipeline fit_predict...")
    start_time = time.time()
    _ = automl.fit_predict(train_data, roles={"target": "TARGET"})
    logging.debug("AutoML pipeline fitted and predicted. Time = {:.3f} sec".format(time.time() - start_time))

    logging.debug(
        "Feature importances of BoostLGBM model. Pay attention to groupby features:\n{}".format(
            automl.levels[0][0].ml_algos[0].get_features_score()
        )
    )
