import logging
import socket
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple, Dict

import pyspark
from pyspark import RDD
from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import SparkSession

VERBOSE_LOGGING_FORMAT = '%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s'

logger = logging.getLogger(__name__)


@contextmanager
def spark_session(session_args: Optional[dict] = None, master: str = "local[]", wait_secs_after_the_end: Optional[int] = None) -> SparkSession:
    """
    Args:
        master: address of the master
            to run locally - "local[1]"

            to run on spark cluster - "spark://node4.bdcl:7077"
            (Optionally set the driver host to a correct hostname .config("spark.driver.host", "node4.bdcl"))

        wait_secs_after_the_end: amount of seconds to wait before stoping SparkSession and thus web UI.

    Returns:
        SparkSession to be used and that is stopped upon exiting this context manager
    """

    if not session_args:
        spark_sess_builder = (
            SparkSession
            .builder
            .appName("SPARK-LAMA-app")
            .master(master)
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.driver.cores", "4")
            .config("spark.driver.memory", "16g")
            .config("spark.cores.max", "16")
            .config("spark.executor.instances", "4")
            .config("spark.executor.memory", "16g")
            .config("spark.executor.cores", "4")
            .config("spark.memory.fraction", "0.6")
            .config("spark.memory.storageFraction", "0.5")
            .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        )
    else:
        spark_sess_builder = (
            SparkSession
            .builder
            .appName("SPARK-LAMA-app")
        )
        for arg, value in session_args.items():
            spark_sess_builder = spark_sess_builder.config(arg, value)

        if 'spark.master' in session_args and not session_args['spark.master'].startswith('local['):
            local_ip_address = socket.gethostbyname(socket.gethostname())
            logger.info(f"Using IP address for spark driver: {local_ip_address}")
            spark_sess_builder = spark_sess_builder.config('spark.driver.host', local_ip_address)

    spark_sess = spark_sess_builder.getOrCreate()

    logger.info(f"Spark WebUI url: {spark_sess.sparkContext.uiWebUrl}")

    try:
        yield spark_sess
    finally:
        logger.info(f"The session is ended. Sleeping {wait_secs_after_the_end if wait_secs_after_the_end else 0} "
                    f"secs until stop the spark session.")
        if wait_secs_after_the_end:
            time.sleep(wait_secs_after_the_end)
        spark_sess.stop()


@contextmanager
def log_exec_time(name: Optional[str] = None, write_log=True):

    # Add file handler for INFO
    if write_log:
        file_handler_info = logging.FileHandler(f'/tmp/{name}_log.log.log', mode='a')
        file_handler_info.setFormatter(logging.Formatter('%(message)s'))
        file_handler_info.setLevel(logging.INFO)
        logger.addHandler(file_handler_info)

    start = datetime.now()

    yield 

    end = datetime.now()
    duration = (end - start).total_seconds()

    msg = f"Exec time of {name}: {duration}" if name else f"Exec time: {duration}"
    logger.warning(msg)


# log_exec_time() class to return elapsed time value
class log_exec_timer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._start = None
        self._duration = None

    def __enter__(self):
        self._start = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self._duration = (datetime.now() - self._start).total_seconds()
        msg = f"Exec time of {self.name}: {self._duration}" if self.name else f"Exec time: {self._duration}"
        logger.info(msg)

    @property
    def duration(self):
        return self._duration


SparkDataFrame = pyspark.sql.DataFrame


def get_cached_df_through_rdd(df: SparkDataFrame, name: Optional[str] = None) -> Tuple[SparkDataFrame, RDD]:
    rdd = df.rdd
    cached_rdd = rdd.setName(name).cache() if name else rdd.cache()
    cached_df = df.sql_ctx.createDataFrame(cached_rdd, df.schema)
    return cached_df, cached_rdd


def logging_config(level: int = logging.INFO, log_filename: str = '/var/log/lama.log') -> dict:
    return {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'verbose': {
                'format': VERBOSE_LOGGING_FORMAT
            },
            'simple': {
                'format': '%(asctime)s %(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'verbose'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'formatter': 'verbose',
                'filename': log_filename
            }
        },
        'loggers': {
            'lightautoml': {
                'handlers': ['console', 'file'],
                'propagate': True,
                'level': level,
            },
            'lightautoml.spark': {
                'handlers': ['console', 'file'],
                'level': level,
                'propagate': False,
            },
            'lightautoml.ml_algo': {
                'handlers': ['console', 'file'],
                'level': level,
                'propagate': False,
            }
        }
    }


def cache(df: SparkDataFrame) -> SparkDataFrame:
    if not df.is_cached:
        df = df.cache()
    return df


def warn_if_not_cached(df: SparkDataFrame):
    if not df.is_cached:
        warnings.warn("Attempting to calculate shape on not cached dataframe. "
                      "It may take too much time.", RuntimeWarning)



class NoOpTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name

    def _transform(self, dataset):
        return dataset


class DebugTransformer(Transformer):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name

    def _transform(self, dataset):
        dataset = dataset.cache()
        dataset.write.mode('overwrite').format('noop').save()
        return dataset


class Cacher(Estimator):
    _cacher_dict: Dict[str, SparkDataFrame] = dict()

    @classmethod
    def get_dataset_by_key(cls, key: str) -> Optional[SparkDataFrame]:
        return cls._cacher_dict.get(key, None)

    @classmethod
    def release_cache_by_key(cls, key: str):
        df = cls._cacher_dict.pop(key, None)
        if df is not None:
            df.unpersist()
            del df

    @property
    def dataset(self) -> SparkDataFrame:
        """Returns chached dataframe"""
        return self._cacher_dict[self._key]

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._dataset: Optional[SparkDataFrame] = None

    def _fit(self, dataset):
        logger.info(f"Cacher {self._key} (RDD Id: {dataset.rdd.id()}). Starting to materialize data.")
        ds = dataset.localCheckpoint(eager=True)
        logger.info(f"Cacher {self._key} (RDD Id: {ds.rdd.id()}). Finished data materialization.")

        previous_ds = self._cacher_dict.get(self._key, None)
        if previous_ds is not None:
            logger.info(f"Removing cache for key: {self._key} (RDD Id: {previous_ds.rdd.id()}).")
            previous_ds.unpersist()
            del previous_ds

        self._cacher_dict[self._key] = ds

        return NoOpTransformer(name=f"cacher_{self._key}")


class EmptyCacher(Cacher):
    def __init__(self, key: str):
        super().__init__(key)
        self._dataset: Optional[SparkDataFrame] = None

    @property
    def dataset(self) -> SparkDataFrame:
        return self._dataset

    def _fit(self, dataset):
        self._dataset = dataset
        return NoOpTransformer(name=f"empty_cacher_{self._key}")
