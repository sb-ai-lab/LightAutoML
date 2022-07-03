# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging.config
import math
import os
import pickle
import random
import shutil
import time
import uuid
from contextlib import contextmanager
from copy import copy
from typing import Dict, Any, List, Optional, Tuple, cast

import yaml
from pyspark import SparkFiles
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.pandas.functions import pandas_udf
from synapse.ml.lightgbm import LightGBMRegressor

from dataset_utils import datasets
from lightautoml.dataset.roles import CategoryRole
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator, SparkTargetEncoderEstimator, \
    SparkCatIntersectionsEstimator
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT, SparkDataFrame
from lightautoml.spark.validation.iterators import SparkFoldsIterator, SparkDummyIterator

import pandas as pd
import numpy as np

logger = logging.getLogger()

DUMP_METADATA_NAME = "metadata.pickle"
DUMP_DATA_NAME = "data.parquet"


@contextmanager
def open_spark_session() -> Tuple[SparkSession, str]:
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = (
            SparkSession
            .builder
            .getOrCreate()
        )
        config_path = SparkFiles.get('config.yaml')
    else:
        spark_sess = (
            SparkSession
            .builder
            .master("local[4]")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "12g")
            .config("spark.executor.memory", "12g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            # .config("spark.eventLog.enabled", "true")
            # .config("spark.eventLog.dir", "file:///tmp/spark_logs")
            .getOrCreate()
        )
        config_path = '/tmp/config.yaml'

    spark_sess.sparkContext.setLogLevel("WARN")
    spark_sess.sparkContext.setCheckpointDir(f"/tmp/chkp_{uuid.uuid4()}")

    try:
        yield spark_sess, config_path
    finally:
        # wait_secs = 600
        # logger.info(f"Sleeping {wait_secs} secs before stopping")
        # time.sleep(wait_secs)
        spark_sess.stop()
        logger.info("Stopped spark session")


def dump_data(path: str, ds: SparkDataset, **meta_kwargs):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    metadata = {
        "roles": ds.roles,
        "target": ds.target_column,
        "folds": ds.folds_column,
        "task_name": ds.task.name if ds.task else None
    }
    metadata.update(meta_kwargs)

    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    sdf = ds.data
    cols = [F.col(c).alias(c.replace('(', '[').replace(')', ']')) for c in sdf.columns]
    sdf = sdf.select(*cols)
    sdf.write.mode('overwrite').parquet(data_file)


def load_dump_if_exist(spark: SparkSession, path: str) -> Optional[Tuple[SparkDataset, Dict]]:
    if not os.path.exists(path):
        return None

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    ex_instances = int(spark.conf.get('spark.executor.instances'))
    ex_cores = int(spark.conf.get('spark.executor.cores'))

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    df = spark.read.parquet(data_file)
    cols = [F.col(c).alias(c.replace('[', '(').replace(']', ')')) for c in df.columns]
    df = df.select(*cols).repartition(ex_instances * ex_cores).cache()

    df.write.mode('overwrite').format('noop').save()

    ds = SparkDataset(
        data=df,
        roles=metadata["roles"],
        task=SparkTask(metadata["task_name"]),
        target=metadata["target"],
        folds=metadata["folds"]
    )

    return ds, metadata


def prepare_test_and_train(spark: SparkSession, path:str, seed: int, test_proportion: float = 0.2) -> Tuple[SparkDataFrame, SparkDataFrame]:
    assert 0.0 <= test_proportion <= 1.0

    train_proportion = 1.0 - test_proportion

    data = spark.read.csv(path, header=True, escape="\"")

    # ex_instances = int(spark.conf.get('spark.executor.instances'))
    # ex_cores = int(spark.conf.get('spark.executor.cores'))

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    # ).repartition(ex_instances * ex_cores).cache()
    data.write.mode('overwrite').format('noop').save()
    # train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

    train_data = data.where(F.col('is_test') < train_proportion).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= train_proportion).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


def load_and_predict_automl(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        roles: Optional[Dict] = None,
        dataset_increase_factor: int = 1,
        automl_model_path=None,
        test_data_dump_path = None,
        **_) -> Dict[str, Any]:

    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))
    memory = spark.conf.get('spark.executor.memory')

    roles = roles if roles else {}

    # train_data, test_data = prepare_test_and_train(spark, path, seed)
    test_data = spark.read.parquet(test_data_dump_path)
    # test_data = test_data.sample(fraction=0.0002, seed=100)

    if dataset_increase_factor > 1:
        test_data = test_data.withColumn("new_col", F.explode(F.array(*[F.lit(0) for i in range(dataset_increase_factor)])))
        test_data = test_data.drop("new_col")
        test_data = test_data.select(
            *[c for c in test_data.columns if c != SparkDataset.ID_COLUMN],
            F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        ).cache()
        test_data = test_data.repartition(execs * cores, SparkDataset.ID_COLUMN).cache()
        test_data = test_data.cache()
        test_data.write.mode('overwrite').format('noop').save()
        logger.info(f"Duplicated dataset size: {test_data.count()}")

    with log_exec_timer("Loading model time") as loading_timer:
        pipeline_model = PipelineModel.load(automl_model_path)

    with log_exec_timer("spark-lama predicting on test") as predict_timer:
        te_pred = pipeline_model.transform(test_data)
        te_pred = te_pred.cache()
        te_pred.write.mode('overwrite').format('noop').save()

    task = SparkTask(task_type)
    pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
    score = task.get_dataset_metric()
    test_metric_value = score(te_pred.select(
        SparkDataset.ID_COLUMN,
        F.col(roles['target']).alias('target'),
        F.col(pred_column).alias('prediction')
    ))

    logger.info(f"score for test predictions via loaded pipeline: {test_metric_value}")

    return {
        "predict_data.count": test_data.count(),
        "spark.executor.instances": execs,
        "spark.executor.cores": cores,
        "spark.executor.memory": memory,
        "test_metric_value": test_metric_value,
        "predict_duration_secs": predict_timer.duration
    }


def calculate_automl(
        spark: SparkSession,
        path: str,
        task_type: str,
        metric_name: str,
        seed: int = 42,
        cv: int = 5,
        use_algos = ("lgb", "linear_l2"),
        roles: Optional[Dict] = None,
        lgb_num_iterations: int = 100,
        linear_l2_reg_param: List[float] = [1e-5],
        dataset_increase_factor: int = 1,
        automl_save_path = None,
        test_data_dump_path = None,
        **_) -> Dict[str, Any]:
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))

    roles = roles if roles else {}

    train_data, test_data = prepare_test_and_train(spark, path, seed)

    if dataset_increase_factor > 0:
        train_data = train_data.withColumn("new_col", F.explode(F.array(*[F.lit(0) for i in range(dataset_increase_factor)])))
        train_data = train_data.drop("new_col")
        train_data = train_data.repartition(execs * cores).cache()
        # train_data = train_data.cache()
        train_data.write.mode('overwrite').format('noop').save()
        logger.info(f"Duplicated dataset size: {train_data.count()}")

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": use_algos},
            lgb_params={'use_single_dataset_mode': True,
                        "default_params": { "numIterations": lgb_num_iterations, "earlyStoppingRound": 5000}, "freeze_defaults": True },
            linear_l2_params={"default_params": {"regParam": linear_l2_reg_param}},
            reader_params={"cv": cv, "advanced_roles": False},
            gbm_pipeline_params={'max_intersection_depth': 2, 'top_intersections': 2},
            linear_pipeline_params={'max_intersection_depth': 2, 'top_intersections': 2},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600},
            timeout=3600 * 12
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"{metric_name} score for out-of-fold predictions: {metric_value}")

    automl.release_cache()

    with log_exec_timer("spark-lama predicting on test") as predict_timer:
        te_pred = automl.predict(test_data, add_reader_attrs=True)

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

        logger.info(f"{metric_name} score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    if automl_save_path:
        transformer = automl.make_transformer()
        transformer.write().overwrite().save(automl_save_path)

    if test_data_dump_path:
        test_data.write.mode('overwrite').parquet(test_data_dump_path)

    return {
        "use_algos": use_algos,
        "train_data.count": train_data.count(),
        "spark.executor.instances": execs,
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration
    }


def calculate_lgbadv_boostlgb(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    with log_exec_timer("spark-lama ml_pipe") as pipe_timer:
        if checkpoint_path is not None:
            train_checkpoint_path = os.path.join(checkpoint_path, 'train.dump')
            test_checkpoint_path = os.path.join(checkpoint_path, 'test.dump')
            train_chkp = load_dump_if_exist(spark, train_checkpoint_path)
            test_chkp = load_dump_if_exist(spark, test_checkpoint_path)
        else:
            train_checkpoint_path = None
            test_checkpoint_path = None
            train_chkp = None
            test_chkp = None

        task = SparkTask(task_type)

        if not train_chkp or not test_chkp:
            logger.info(f"Checkpoint doesn't exist on path {checkpoint_path}. Will create it.")

            train_data, test_data = prepare_test_and_train(spark, path, seed)

            sreader = SparkToSparkReader(task=task, cv=3, advanced_roles=False)
            sdataset = sreader.fit_read(train_data, roles=roles)

            ml_alg_kwargs = {
                'auto_unique_co': 10,
                'max_intersection_depth': 3,
                'multiclass_te_co': 3,
                'output_categories': True,
                'top_intersections': 4
            }

            lgb_features = SparkLGBAdvancedPipeline(**ml_alg_kwargs)
            lgb_features.input_roles = sdataset.roles
            sdataset = lgb_features.fit_transform(sdataset)

            iterator = SparkFoldsIterator(sdataset, n_folds=cv)
            iterator.input_roles = lgb_features.output_roles

            stest = sreader.read(test_data, add_array_attrs=True)
            stest = cast(SparkDataset, lgb_features.transform(stest))

            if checkpoint_path is not None:
                dump_data(train_checkpoint_path, iterator.train, iterator_input_roles=iterator.input_roles)
                dump_data(test_checkpoint_path, stest, iterator_input_roles=iterator.input_roles)
        else:
            logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")

            train_chkp_ds, metadata = train_chkp

            df = train_chkp_ds.data
            df = df.withColumn("new_col", F.explode(F.array(*[F.lit(0) for i in range(1)])))
            df = df.drop("new_col")
            df = df.cache()
            print(f"Duplicated dataset size: {df.count()}")
            new_train_chkp_ds = train_chkp_ds.empty()
            new_train_chkp_ds.set_data(df, train_chkp_ds.features, train_chkp_ds.roles)
            train_chkp_ds = new_train_chkp_ds

            iterator = SparkFoldsIterator(train_chkp_ds, n_folds=cv)
            iterator.input_roles = metadata['iterator_input_roles']

            stest, _ = test_chkp

        iterator = iterator.convert_to_holdout_iterator()

        score = task.get_dataset_metric()

        with log_exec_timer("Boost_time") as boost_timer:
            spark_ml_algo = SparkBoostLGBM(
                cacher_key='main_cache',
                use_single_dataset_mode=True,
                default_params={
                    "learningRate": 0.05,
                    "numLeaves": 128,
                    "featureFraction": 0.7,
                    "baggingFraction": 0.7,
                    "baggingFreq": 1,
                    "maxDepth": -1,
                    "minGainToSplit": 0.0,
                    "maxBin": 255,
                    "minDataInLeaf": 5,
                    # e.g. num trees
                    "numIterations": 100,
                    "earlyStoppingRound": 5000,
                    # for regression
                    "alpha": 1.0,
                    "lambdaL1": 0.0,
                    "lambdaL2": 0.0
                },
                freeze_defaults=True,
                max_validation_size=10_000
            )
            spark_ml_algo, oof_preds = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)

        assert spark_ml_algo is not None
        assert oof_preds is not None

        spark_ml_algo = cast(SparkTabularMLAlgo, spark_ml_algo)
        oof_preds = cast(SparkDataset, oof_preds)
        oof_preds_sdf = oof_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(oof_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        oof_score = score(oof_preds_sdf)

        test_preds = spark_ml_algo.predict(stest)
        test_preds_sdf = test_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(test_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        test_score = score(test_preds_sdf)

    return {
        pipe_timer.name: pipe_timer.duration,
        boost_timer.name: boost_timer.duration,
        'oof_score': oof_score,
        'test_score': test_score
    }


def calculate_linear_l2(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    with log_exec_timer("spark-lama ml_pipe") as pipe_timer:
        if checkpoint_path is not None:
            train_checkpoint_path = os.path.join(checkpoint_path, 'train.dump')
            test_checkpoint_path = os.path.join(checkpoint_path, 'test.dump')
            train_chkp = load_dump_if_exist(spark, train_checkpoint_path)
            test_chkp = load_dump_if_exist(spark, test_checkpoint_path)
        else:
            train_checkpoint_path = None
            test_checkpoint_path = None
            train_chkp = None
            test_chkp = None

        task = SparkTask(task_type)

        if not train_chkp or not test_chkp:
            logger.info(f"Checkpoint doesn't exist on path {checkpoint_path}. Will create it.")

            train_data, test_data = prepare_test_and_train(spark, path, seed)

            sreader = SparkToSparkReader(task=task, cv=3, advanced_roles=False)
            sdataset = sreader.fit_read(train_data, roles=roles)

            ml_alg_kwargs = {
                'auto_unique_co': 10,
                'max_intersection_depth': 3,
                'multiclass_te_co': 3,
                'output_categories': True,
                'top_intersections': 4
            }

            features = SparkLinearFeatures(**ml_alg_kwargs)
            features.input_roles = sdataset.roles
            sdataset = features.fit_transform(sdataset)

            iterator = SparkFoldsIterator(sdataset, n_folds=cv)
            iterator.input_roles = features.output_roles

            stest = sreader.read(test_data, add_array_attrs=True)
            stest = cast(SparkDataset, features.transform(stest))

            if checkpoint_path is not None:
                dump_data(train_checkpoint_path, iterator.train, iterator_input_roles=iterator.input_roles)
                dump_data(test_checkpoint_path, stest, iterator_input_roles=iterator.input_roles)
        else:
            logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")

            exec_cores = int(spark.conf.get("spark.executor.cores"))
            exec_instances = int(spark.conf.get("spark.executor.instances"))

            train_chkp_ds, metadata = train_chkp

            df = train_chkp_ds.data
            df = df.withColumn("new_col", F.explode(F.array(*[F.lit(0) for i in range(10)])))
            df = df.drop("new_col")
            df = df.repartition(exec_cores * exec_instances).cache()
            print(f"Duplicated dataset size: {df.count()}")
            new_train_chkp_ds = train_chkp_ds.empty()
            new_train_chkp_ds.set_data(df, train_chkp_ds.features, train_chkp_ds.roles)
            train_chkp_ds = new_train_chkp_ds

            iterator = SparkFoldsIterator(train_chkp_ds, n_folds=cv)
            iterator.input_roles = metadata['iterator_input_roles']

            stest, _ = test_chkp

        iterator = iterator.convert_to_holdout_iterator()
        # iterator = SparkDummyIterator(iterator.train, iterator.input_roles)

        score = task.get_dataset_metric()

        with log_exec_timer("Linear_time") as linear_timer:
            spark_ml_algo = SparkLinearLBFGS(
                cacher_key='main_cache',
                default_params={"regParam": [1e-5]},
                freeze_defaults=True
            )
            spark_ml_algo, oof_preds = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)

        assert spark_ml_algo is not None
        assert oof_preds is not None

        spark_ml_algo = cast(SparkTabularMLAlgo, spark_ml_algo)
        oof_preds = cast(SparkDataset, oof_preds)
        oof_preds_sdf = oof_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(oof_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        oof_score = score(oof_preds_sdf)

        test_preds = spark_ml_algo.predict(stest)
        test_preds_sdf = test_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(test_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        test_score = score(test_preds_sdf)

    return {
        pipe_timer.name: pipe_timer.duration,
        linear_timer.name: linear_timer.duration,
        'oof_score': oof_score,
        'test_score': test_score
    }


def calculate_reader(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        **_):

    data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

    task = SparkTask(task_type)

    with log_exec_timer("Reader") as reader_timer:
        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(data, roles=roles)
        sdataset.data.write.mode('overwrite').format('noop').save()

    return {"reader_time": reader_timer.duration}


def calculate_le(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        **_):

    data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

    task = SparkTask(task_type)

    with log_exec_timer("Reader") as reader_timer:
        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(data, roles=roles)

    cat_roles = {feat: role for feat, role in sdataset.roles.items() if isinstance(role, CategoryRole)}

    with log_exec_timer("SparkLabelEncoder") as le_timer:
        estimator = SparkLabelEncoderEstimator(
            input_cols=list(cat_roles.keys()),
            input_roles=cat_roles
        )

        transformer = estimator.fit(data)

    with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
        df = transformer.transform(data)
        df.write.mode('overwrite').format('noop').save()

    return {
        "reader_time": reader_timer.duration,
        "le_fit_time": le_timer.duration,
        "le_transform_time": le_transform_timer.duration
    }


def calculate_te(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        dataset_increase_factor: int = 1,
        **_):

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'data.dump')
        chkp = load_dump_if_exist(spark, checkpoint_path)
    else:
        checkpoint_path = None
        chkp = None

    task = SparkTask(task_type)

    if not chkp:
        data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

        execs = int(spark.conf.get('spark.executor.instances'))
        cores = int(spark.conf.get('spark.executor.cores'))

        data = data.withColumn("new_col",
                               F.explode(F.array(*[F.lit(0) for i in range(dataset_increase_factor)])))
        data = data.drop("new_col")
        data = data.repartition(execs * cores).cache()
        data = data.cache()
        data.write.mode('overwrite').format('noop').save()
        print(f"Duplicated dataset size: {data.count()}")

        task = SparkTask(task_type)

        with log_exec_timer("Reader") as reader_timer:
            sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
            sdataset = sreader.fit_read(data, roles=roles)

        cat_roles = {feat: role for feat, role in sdataset.roles.items() if isinstance(role, CategoryRole)}

        with log_exec_timer("SparkLabelEncoder") as le_timer:
            estimator = SparkLabelEncoderEstimator(
                input_cols=list(cat_roles.keys()),
                input_roles=cat_roles
            )

            transformer = estimator.fit(sdataset.data)

        with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
            df = transformer.transform(sdataset.data).localCheckpoint(eager=True)

        df = df.select(
            SparkDataset.ID_COLUMN,
            sdataset.folds_column,
            sdataset.target_column,
            *estimator.getOutputCols()
        )
        le_ds = sdataset.empty()
        le_ds.set_data(df, estimator.getOutputCols(), estimator.getOutputRoles())

        if checkpoint_path is not None:
            dump_data(checkpoint_path, le_ds)
    else:
        logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")
        le_ds, _ = chkp

    with log_exec_timer("TargetEncoder") as te_timer:
        te_estimator = SparkTargetEncoderEstimator(
            input_cols=le_ds.features,
            input_roles=le_ds.roles,
            task_name=task.name,
            folds_column=le_ds.folds_column,
            target_column=le_ds.target_column
        )

        te_transformer = te_estimator.fit(le_ds.data)

    with log_exec_timer("TargetEncoder transform") as te_transform_timer:
        df = te_transformer.transform(le_ds.data)
        df.write.mode('overwrite').format('noop').save()

    if not chkp:
        res = {
            "reader_time": reader_timer.duration,
            "le_fit_time": le_timer.duration,
            "le_transform_time": le_transform_timer.duration,
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }
    else:
        res = {
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }

    return res


def calculate_cat_te(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        dataset_increase_factor: int = 1,
        **_):

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'data.dump')
        chkp = load_dump_if_exist(spark, checkpoint_path)
    else:
        checkpoint_path = None
        chkp = None

    task = SparkTask(task_type)

    if not chkp:
        data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

        execs = int(spark.conf.get('spark.executor.instances'))
        cores = int(spark.conf.get('spark.executor.cores'))

        data = data.withColumn("new_col",
                               F.explode(F.array(*[F.lit(0) for i in range(dataset_increase_factor)])))
        data = data.drop("new_col")
        data = data.repartition(execs * cores).cache()
        data = data.cache()
        data.write.mode('overwrite').format('noop').save()
        print(f"Duplicated dataset size: {data.count()}")

        task = SparkTask(task_type)

        with log_exec_timer("Reader") as reader_timer:
            sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
            sdataset = sreader.fit_read(data, roles=roles)

        cat_roles = {feat: role for feat, role in sdataset.roles.items() if feat in ['vin', 'city', 'power', 'torque']}

        with log_exec_timer("SparkLabelEncoder") as ci_timer:
            estimator = SparkCatIntersectionsEstimator(
                input_cols=list(cat_roles.keys()),
                input_roles=cat_roles,
                max_depth=2
            )

            transformer = estimator.fit(sdataset.data)

        with log_exec_timer("SparkLabelEncoder transform") as ci_transform_timer:
            df = transformer.transform(sdataset.data)
            # df.write.mode('overwrite').format('noop').save()
            df = cast(SparkDataFrame, df)
            df = df.localCheckpoint(eager=True)

        df = df.select(
            SparkDataset.ID_COLUMN,
            sdataset.folds_column,
            sdataset.target_column,
            *estimator.getOutputCols()
        )
        le_ds = sdataset.empty()
        le_ds.set_data(df, estimator.getOutputCols(), estimator.getOutputRoles())

        if checkpoint_path is not None:
            dump_data(checkpoint_path, le_ds)
    else:
        logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")
        le_ds, _ = chkp

    with log_exec_timer("Intermediate test") as f:
        for _ in range(3):
            le_ds.data.select([F.col(c) * 2 for c in le_ds.data.columns]).write.mode('overwrite').format('noop').save()

    print(f"INTERMEDIATE TEST DURATION: {f.duration}")

    with log_exec_timer("TargetEncoder") as te_timer:
        te_estimator = SparkTargetEncoderEstimator(
            input_cols=le_ds.features,
            input_roles=le_ds.roles,
            task_name=task.name,
            folds_column=le_ds.folds_column,
            target_column=le_ds.target_column
        )

        te_transformer = te_estimator.fit(le_ds.data)

    with log_exec_timer("TargetEncoder transform") as te_transform_timer:
        df = te_transformer.transform(le_ds.data)
        df.write.mode('overwrite').format('noop').save()

    if not chkp:
        res = {
            "reader_time": reader_timer.duration,
            "cat_fit_time": ci_timer.duration,
            "cat_transform_time": ci_transform_timer.duration,
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }
    else:
        res = {
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }

    return res


def empty_calculate(spark: SparkSession, **_):
    logger.info("Success")
    return {"result": "success"}


def calculate_broadcast(spark: SparkSession, **_):
    spark.sparkContext.setCheckpointDir("/tmp/chkp")

    data = [
        {"a": i, "b": i * 10, "c": i * 100}
        for i in range(100)
    ]

    df = spark.createDataFrame(data)
    df = df.cache()
    df.write.mode('overwrite').format('noop').save()

    with log_exec_timer("a") as gen_arr:
        mapping_size = 10_000_000
        bdata = {i: random.randint(0, 1000) for i in range(mapping_size)}

    print(f"Gen arr time: {gen_arr.duration}")

    with log_exec_timer("b") as bcast_timer:
        bval = spark.sparkContext.broadcast(bdata)

    print(f"Bcast time: {bcast_timer.duration}")

    @pandas_udf('int')
    def func1(col: pd.Series) -> pd.Series:
        mapping = bval.value
        # mapping = bdata

        return col.apply(lambda x: x + mapping[x] if x in mapping else 0.0)
        # return col.apply(lambda x: x + 10.0)

    df_1 = df.select([func1(c).alias(c) for c in df.columns])
    df_1 = df_1.cache()
    df_1.write.mode('overwrite').format('noop').save()

    df_1 = df_1.localCheckpoint(eager=True)

    bval.destroy()

    # df_1 = spark.createDataFrame(df_1.rdd, schema=df_1.schema, verifySchema=False)

    # df_1 = spark.createDataFrame(df_1.rdd)

    # with log_exec_timer("b") as chkp_timer:
    #     df_1 = df_1.checkpoint(eager=True)
    #
    # print(f"checkpoint time: {chkp_timer.duration}")

    @pandas_udf('int')
    def func2(col: pd.Series) -> pd.Series:
        return col.apply(lambda x: x - 10)

    df_2 = df_1#df_1.select([func2(c).alias(c) for c in df_1.columns])
    # df_2 = df_2.cache()
    df_2.write.mode('overwrite').format('noop').save()

    print("Finished")


def calculate_le_scaling(spark: SparkSession, path: str, **_):
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))

    df = spark.read.json(path).coalesce(execs * cores).cache()
    df.write.mode('overwrite').format('noop').save()

    cat_roles = {
       c: CategoryRole(dtype=np.float32) for c in df.columns
    }

    with log_exec_timer("SparkLabelEncoder") as le_timer:
        estimator = SparkLabelEncoderEstimator(
            input_cols=list(cat_roles.keys()),
            input_roles=cat_roles
        )

        transformer = estimator.fit(df)

    # with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
    #     df = transformer.transform(df).cache()
    #     df.write.mode('overwrite').format('noop').save()
    import time
    time.sleep(600)

    return {
        "le_fit": le_timer.duration,
        # "le_transform": le_transform_timer.duration
    }


def calculate_le_te_scaling(
        spark: SparkSession,
        path: str,
        checkpoint_path: Optional[str] = None,
        **_):
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'data.dump')
        chkp = load_dump_if_exist(spark, checkpoint_path)
    else:
        checkpoint_path = None
        chkp = None

    if not chkp:
        logger.info(f"No checkpoint found on path {checkpoint_path}. Will create it ")
        data, _ = prepare_test_and_train(spark, path, 42, test_proportion=0.0)
        df = data

        cat_roles = {
           c: CategoryRole(dtype=np.float32) for c in df.columns
        }

        with log_exec_timer("SparkLabelEncoder") as le_timer:
            estimator = SparkLabelEncoderEstimator(
                input_cols=list(cat_roles.keys()),
                input_roles=cat_roles
            )

            transformer = estimator.fit(df)

        cv = 5
        with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
            df = transformer.transform(df).select(
                F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
                F.rand(42).alias('target'),
                F.floor(F.rand(142) * cv).astype('int').alias('folds'),
                *list(estimator.getOutputRoles().keys()),
            ).localCheckpoint(eager=True)
            df.write.mode('overwrite').format('noop').save()

        le_ds = SparkDataset(
            data=df,
            roles=estimator.getOutputRoles(),
            task=SparkTask('reg'),
            target='target',
            folds='folds'
        )

        if checkpoint_path is not None:
            dump_data(checkpoint_path, le_ds)

        result_le = {
            "le_fit": le_timer.duration,
            "le_transform": le_transform_timer.duration,
        }
    else:
        logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")
        result_le = dict()
        le_ds, _ = chkp

    with log_exec_timer("TargetEncoder") as te_timer:
        te_estimator = SparkTargetEncoderEstimator(
            input_cols=le_ds.features,
            input_roles=le_ds.roles,
            task_name='reg',
            folds_column=le_ds.folds_column,
            target_column=le_ds.target_column
        )

        te_transformer = te_estimator.fit(df)

    with log_exec_timer("TargetEncoder transform") as te_transform_timer:
        df = te_transformer.transform(df)
        df.write.mode('overwrite').format('noop').save()

    return {
        **result_le,
        "te_fit": te_timer.duration,
        "te_transform": te_transform_timer.duration
    }


def calculate_le_model_scaling(
        spark: SparkSession,
        path: str,
        model_type: str,
        checkpoint_path: Optional[str] = None,
        **_):
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'data.dump')
        chkp = load_dump_if_exist(spark, checkpoint_path)
    else:
        checkpoint_path = None
        chkp = None

    if not chkp:
        logger.info(f"No checkpoint found on path {checkpoint_path}. Will create it ")
        df = spark.read.json(path).repartition(execs * cores).cache()
        df.write.mode('overwrite').format('noop').save()

        cat_roles = {
           c: CategoryRole(dtype=np.float32) for c in df.columns
        }

        with log_exec_timer("SparkLabelEncoder") as le_timer:
            estimator = SparkLabelEncoderEstimator(
                input_cols=list(cat_roles.keys()),
                input_roles=cat_roles
            )

            transformer = estimator.fit(df)

        cv = 5
        with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
            df = transformer.transform(df).select(
                F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
                F.rand(42).alias('target'),
                F.floor(F.rand(142) * cv).astype('int').alias('folds'),
                *list(estimator.getOutputRoles().keys()),
            ).cache()
            df.write.mode('overwrite').format('noop').save()

        le_ds = SparkDataset(
            data=df,
            roles=estimator.getOutputRoles(),
            task=SparkTask('reg'),
            target='target',
            folds='folds'
        )

        if checkpoint_path is not None:
            dump_data(checkpoint_path, le_ds)

        result_le = {
            "le_fit": le_timer.duration,
            "le_transform": le_transform_timer.duration,
        }
    else:
        logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")
        result_le = dict()
        le_ds, _ = chkp

    assembler = VectorAssembler(
        inputCols=le_ds.features,
        outputCol="assembler_features"
    )

    if model_type == 'linreg':
        model = LinearRegression(featuresCol=assembler.getOutputCol(),
                                   labelCol=le_ds.target_column,
                                   predictionCol='prediction',
                                   maxIter=1000,
                                   aggregationDepth=2,
                                   elasticNetParam=0.7)
    else:
        count = le_ds.data.count()
        num_threads = max(int(spark.conf.get("spark.executor.cores", "1")) - 1, 1)
        num_threads = 7
        model = LightGBMRegressor(
            numThreads=num_threads,
            objective="regression",
            metric="mse",
            learningRate=0.05,
            numLeaves=128,
            # numLeaves=32,
            featureFraction=1.0,
            baggingFraction=1.0,
            baggingFreq=1,
            maxDepth=-1,
            minGainToSplit=0.0,
            maxBin=255,
            minDataInLeaf=5,
            numIterations=100,
            earlyStoppingRound=1000,
            alpha=1.0,
            lambdaL1=0.5,
            lambdaL2=0.0,
            featuresCol=assembler.getOutputCol(),
            labelCol=le_ds.target_column,
            verbosity=1,
            useSingleDatasetMode=True,
            isProvideTrainingMetric=True,
            chunkSize=math.ceil(count / num_threads)
            # chunkSize=700_000
        )

    pipeline = Pipeline(stages=[assembler, model])

    with log_exec_timer("LinReg fit") as fit_timer:
        transformer = pipeline.fit(le_ds.data)

    # with log_exec_timer("LinReg fit") as transform_timer:
    #     df = transformer.transform(le_ds.data)
    #     df.write.mode('overwrite').format('noop').save()

    return {
        **result_le,
        "linreg_fit": fit_timer.duration,
        # "linreg_transform": transform_timer.duration
    }


def calculate_chkp(spark: SparkSession, path: str, **_):
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))

    df = spark.read.csv(path)
    df = df.withColumn("new_col", F.explode(F.array(*[F.lit(0) for i in range(10)])))
    df = df.drop("new_col")
    # df = df.repartition(execs * cores).cache()
    df = df.cache()
    df.write.mode('overwrite').format('noop').save()
    print(f"Duplicated dataset size: {df.count()}")

    with log_exec_timer('chkp-timer') as chkp_timer:
        df.localCheckpoint(eager=True)

    print(f"Chkp time: {chkp_timer.duration}")

    import time
    time.sleep(600)

    return {
        chkp_timer.name: chkp_timer.duration
    }



if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename="/tmp/lama.log"))
    logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)

    with open_spark_session() as (spark, config_path):
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
        elif func_name == "calculate_lgbadv_boostlgb":
            func = calculate_lgbadv_boostlgb
        elif func_name == "calculate_linear_l2":
            func = calculate_linear_l2
        elif func_name == 'empty_calculate':
            func = empty_calculate
        elif func_name == 'calculate_reader':
            func = calculate_reader
        elif func_name == 'calculate_le':
            func = calculate_le
        elif func_name == 'calculate_te':
            func = calculate_te
        elif func_name == 'calculate_cat_te':
            func = calculate_cat_te
        elif func_name == 'calculate_broadcast':
            func = calculate_broadcast
        elif func_name == 'calculate_le_scaling':
            func = calculate_le_scaling
        elif func_name == 'calculate_le_te_scaling':
            func = calculate_le_te_scaling
        elif func_name == 'calculate_le_model_scaling':
            func = calculate_le_model_scaling
        elif func_name == 'calculate_chkp':
            func = calculate_chkp
        elif func_name == 'load_and_predict_automl':
            func = load_and_predict_automl
        else:
            raise ValueError(f"Incorrect func name: {func_name}. ")

        result = func(spark=spark, **ds_cfg)
        print(f"EXP-RESULT: {result}")
        ex_instances = int(spark.conf.get('spark.executor.instances'))
        print(f"spark.executor.instances: {ex_instances}")
