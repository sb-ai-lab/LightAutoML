import logging.config
import os
import uuid

import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from examples_utils import get_dataset_attrs, prepare_test_and_train, get_spark_session
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


def main(spark: SparkSession, dataset_name: str, seed: int):
    # Algos and layers to be used during automl:
    # For example:
    # 1. use_algos = [["lgb"]]
    # 2. use_algos = [["lgb_tuned"]]
    # 3. use_algos = [["linear_l2"]]
    # 4. use_algos = [["lgb", "linear_l2"], ["lgb"]]
    use_algos = [["lgb", "linear_l2"], ["lgb"]]
    cv = 5
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)
        train_data, test_data = prepare_test_and_train(spark, path, seed)

        test_data_dropped = test_data

        # optionally: set 'convert_to_onnx': True to use onnx-based version of lgb's model transformer
        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": use_algos},
            lgb_params={'use_single_dataset_mode': True, 'convert_to_onnx': False, 'mini_batch_size': 1000},
            reader_params={"cv": cv, "advanced_roles": False}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"score for out-of-fold predictions: {metric_value}")

    transformer = automl.make_transformer()

    automl.release_cache()

    with log_exec_timer("spark-lama predicting on test (#1 way)") as predict_timer:
        te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

        logger.info(f"score for test predictions: {test_metric_value}")

    with log_exec_timer("spark-lama predicting on test (#2 way)"):
        te_pred = automl.make_transformer().transform(test_data_dropped)

        pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
        score = task.get_dataset_metric()
        test_metric_value = score(te_pred.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(pred_column).alias('prediction')
        ))

        logger.info(f"score for test predictions: {test_metric_value}")

    base_path = "/tmp/spark_results"
    automl_model_path = os.path.join(base_path, "automl_pipeline")
    os.makedirs(base_path, exist_ok=True)

    with log_exec_timer("saving model") as saving_timer:
        transformer.write().overwrite().save(automl_model_path)

    with log_exec_timer("Loading model time") as loading_timer:
        pipeline_model = PipelineModel.load(automl_model_path)

    with log_exec_timer("spark-lama predicting on test (#3 way)"):
        te_pred = pipeline_model.transform(test_data_dropped)

        pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
        score = task.get_dataset_metric()
        test_metric_value = score(te_pred.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(pred_column).alias('prediction')
        ))

    logger.info(f"score for test predictions via loaded pipeline: {test_metric_value}")

    logger.info("Predicting is finished")

    result = {
        "seed": seed,
        "dataset": dataset_name,
        "used_algo": str(use_algos),
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration,
        "saving_duration_secs": saving_timer.duration,
        "loading_duration_secs": loading_timer.duration
    }

    print(f"EXP-RESULT: {result}")

    train_data.unpersist()
    test_data.unpersist()

    return result


def multirun(spark: SparkSession, dataset_name: str):
    seeds = [1, 5, 42, 100, 777]
    results = [main(spark, dataset_name, seed) for seed in seeds]

    df = pd.DataFrame(results)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    df.to_csv(f"spark-lama_results_{dataset_name}_{uuid.uuid4()}.csv")


if __name__ == "__main__":
    spark_sess = get_spark_session()
    # One can run:
    # 1. main(dataset_name="lama_test_dataste", seed=42)
    # 2. multirun(spark_sess, dataset_name="lama_test_dataset")
    main(spark_sess, dataset_name="lama_test_dataset", seed=42)

    spark_sess.stop()
