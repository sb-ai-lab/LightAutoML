import logging.config

from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.report import SparkReportDeco
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import log_exec_timer
from lightautoml.spark.utils import logging_config

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 5
    use_algos = [["lgb"]]
    dataset_name = "lama_test_dataset"
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    train_data, test_data = prepare_test_and_train(spark, path, seed)

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 3000}},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False, 'random_state': seed}
        )

        report_automl = SparkReportDeco(
            output_path="/tmp/spark",
            report_file_name="spark_lama_report.html",
            interpretation=True
        )(automl)

        report_automl.fit_predict(train_data, roles=roles)
        report_automl.predict(test_data, add_reader_attrs=True)
