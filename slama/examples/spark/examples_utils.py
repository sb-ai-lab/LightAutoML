import os
from typing import Tuple

from pyspark.sql import SparkSession

from lightautoml.spark.utils import SparkDataFrame

used_cars_params = {
    "task_type": "reg",
    "roles": {
        "target": "price",
        "drop": ["dealer_zip", "description", "listed_date",
                 "year", 'Unnamed: 0', '_c0',
                 'sp_id', 'sp_name', 'trimId',
                 'trim_name', 'major_options', 'main_picture_url',
                 'interior_color', 'exterior_color'],
        "numeric": ['latitude', 'longitude', 'mileage']
    },
    "dtype": {
        'fleet': 'str', 'frame_damaged': 'str',
        'has_accidents': 'str', 'isCab': 'str',
        'is_cpo': 'str', 'is_new': 'str',
        'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
    }
}

DATASETS = {
    "used_cars_dataset": {
            "path": "file:///opt/spark_data/small_used_cars_data.csv",
            **used_cars_params
    },

    "used_cars_dataset_1x": {
        "path": "file:///opt/spark_data/derivative_datasets/1x_dataset.csv",
        **used_cars_params
    },

    "used_cars_dataset_4x": {
        "path": "file:///opt/spark_data/derivative_datasets/4x_dataset.csv",
        **used_cars_params
    },

    # https://www.openml.org/d/4549
    "buzz_dataset": {
        "path": "file:///opt/spark_data/Buzzinsocialmedia_Twitter_25k.csv",
        "task_type": "reg",
        "roles": {"target": "Annotation"},
    },

    "lama_test_dataset": {
        "path": "file:///opt/spark_data/sampled_app_train.csv",
        "task_type": "binary",
        "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    },

    # https://www.openml.org/d/734
    "ailerons_dataset": {
        "path": "file:///opt/spark_data/ailerons.csv",
        "task_type": "binary",
        "roles": {"target": "binaryClass"},
    },

    # https://www.openml.org/d/382
    "ipums_97": {
        "path": "file:///opt/spark_data/ipums_97.csv",
        "task_type": "multiclass",
        "roles": {"target": "movedin"},
    },

    "company_bankruptcy_dataset": {
        "path": "file:///opt/spark_data/company_bankruptcy_prediction_data.csv",
        "task_type": "binary",
        "roles": {"target": "Bankrupt?"},
    }
}


def get_dataset_attrs(name: str):
    return (
        DATASETS[name]['path'],
        DATASETS[name]['task_type'],
        DATASETS[name]['roles'],
        # to assure that LAMA correctly interprets certain columns as categorical
        DATASETS[name].get('dtype', dict()),
    )


def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")
    data = data.cache()
    data.write.mode('overwrite').format('noop').save()

    train_data, test_data = data.randomSplit([0.8, 0.2], seed)
    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    data.unpersist()

    return train_data, test_data


def get_spark_session():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
    else:
        spark_sess = (
            SparkSession
            .builder
            .master("local[4]")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            .config("spark.cleaner.referenceTracking", "true")
            .config("spark.cleaner.periodicGC.interval", "1min")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "16g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

    spark_sess.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    spark_sess.sparkContext.setLogLevel("WARN")

    return spark_sess
