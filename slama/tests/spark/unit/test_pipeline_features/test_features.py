import logging.config
import logging.config
import os
from typing import Dict, Any, cast, Optional

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.spark.validation.iterators import SparkFoldsIterator
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import FoldsIterator
from .. import spark as spark_sess
from ..dataset_utils import get_test_datasets, prepared_datasets, load_dump_if_exist, dump_data

spark = spark_sess

# DATASETS_ARG = {"dataset": "lama_test_dataset"}
DATASETS_ARG = {"dataset": "used_cars_dataset"}
# DATASETS_ARG = {"setting": "binary"}

CV = 5
seed = 42

# otherwise holdout is used
USE_FOLDS_VALIDATION = True

ml_alg_kwargs = {
    'auto_unique_co': 10,
    'max_intersection_depth': 3,
    'multiclass_te_co': 3,
    'output_categories': True,
    'top_intersections': 4
}


logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def compare_feature_pipelines_by_quality(spark: SparkSession, cv: int, config: Dict[str, Any],
                                         fp_lama_clazz, ml_algo_lama_clazz,
                                         ml_alg_kwargs: Dict[str, Any], pipeline_name: str):
    checkpoint_dir = '/opt/test_checkpoints/feature_pipelines'
    path = config['path']
    ds_name = os.path.basename(os.path.splitext(path)[0])

    dump_train_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_train.dump") \
        if checkpoint_dir is not None else None
    dump_test_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_test.dump") \
        if checkpoint_dir is not None else None

    train_res = load_dump_if_exist(spark, dump_train_path)
    test_res = load_dump_if_exist(spark, dump_test_path)
    if not train_res or not test_res:
        raise ValueError("Dataset should be processed with feature pipeline "
                         "and the corresponding dump should exist. Please, run corresponding non-quality test first.")
    dumped_train_ds, _ = train_res
    dumped_test_ds, _ = test_res

    test_ds = dumped_test_ds.to_pandas() if ml_algo_lama_clazz == BoostLGBM else dumped_test_ds.to_pandas().to_numpy()

    # Process spark-based features with LAMA
    pds = dumped_train_ds.to_pandas() if ml_algo_lama_clazz == BoostLGBM else dumped_train_ds.to_pandas().to_numpy()

    train_valid = FoldsIterator(pds)
    ml_algo = ml_algo_lama_clazz()
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    assert ml_algo is not None
    test_pred = ml_algo.predict(test_ds)
    score = train_valid.train.task.get_dataset_metric()
    spark_based_oof_metric = score(oof_pred)
    spark_based_test_metric = score(test_pred)

    # compare with native features of LAMA
    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    train_pdf = pd.read_csv(config['train_path'], **read_csv_args)
    test_pdf = pd.read_csv(config['test_path'], **read_csv_args)
    # train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=100)
    reader = PandasToPandasReader(task=Task(train_valid.train.task.name), cv=cv, advanced_roles=False, random_state=seed)
    train_ds = reader.fit_read(train_pdf, roles=config['roles'])
    test_ds = reader.read(test_pdf, add_array_attrs=True)
    lama_pipeline = fp_lama_clazz(**ml_alg_kwargs)
    lama_feats = lama_pipeline.fit_transform(train_ds)
    lama_test_feats = lama_pipeline.transform(test_ds)
    lama_feats = lama_feats if ml_algo_lama_clazz == BoostLGBM else lama_feats.to_numpy()

    train_valid = FoldsIterator(lama_feats.to_numpy())
    ml_algo = ml_algo_lama_clazz()
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    assert ml_algo is not None
    test_pred = ml_algo.predict(lama_test_feats)
    # test_pred = ml_algo.predict(test_ds)
    score = train_valid.train.task.get_dataset_metric()
    lama_oof_metric = score(oof_pred)
    lama_test_metric = score(test_pred)

    print(f"LAMA oof: {lama_oof_metric}. LAMA-on-Spark oof: {spark_based_oof_metric}")
    print(f"LAMA test: {lama_test_metric}. LAMA-on-Spark test: {spark_based_test_metric}")

    max_diff_in_percents = 0.05

    # assert spark_based_oof_metric > lama_oof_metric or abs((lama_oof_metric - spark_based_oof_metric) / max(lama_oof_metric, spark_based_oof_metric)) < max_diff_in_percents
    # assert spark_based_oof_metric > lama_oof_metric or abs((lama_oof_metric - spark_based_oof_metric) / min(lama_oof_metric, spark_based_oof_metric)) < max_diff_in_percents

    assert spark_based_test_metric > lama_test_metric or abs((lama_test_metric - spark_based_test_metric) / max(lama_test_metric, spark_based_test_metric)) < max_diff_in_percents
    assert spark_based_test_metric > lama_test_metric or abs((lama_test_metric - spark_based_test_metric) / min(lama_test_metric, spark_based_test_metric)) < max_diff_in_percents


def compare_feature_pipelines(spark: SparkSession, cv: int, ds_config: Dict[str, Any],
                              lama_clazz, slama_clazz, ml_alg_kwargs: Dict[str, Any], pipeline_name: str):
    checkpoint_fp_dir = '/opt/test_checkpoints/feature_pipelines'
    spark_dss = prepared_datasets(spark, cv, [ds_config], checkpoint_dir='/opt/test_checkpoints/reader_datasets')
    spark_train_ds, spark_test_ds = spark_dss[0]

    ds_name = os.path.basename(os.path.splitext(ds_config['path'])[0])

    # LAMA pipeline
    read_csv_args = {'dtype':  ds_config['dtype']} if 'dtype' in ds_config else dict()
    pdf = pd.read_csv(ds_config['train_path'], **read_csv_args)
    reader = PandasToPandasReader(task=Task(spark_train_ds.task.name), cv=cv, advanced_roles=False)
    ds = reader.fit_read(pdf, roles=ds_config['roles'])

    lama_pipeline = lama_clazz(**ml_alg_kwargs)
    lama_feats = lama_pipeline.fit_transform(ds)
    lf_pds = cast(PandasDataset, lama_feats.to_pandas())

    # SLAMA pipeline
    slama_pipeline = slama_clazz(**ml_alg_kwargs)
    slama_pipeline.input_roles = spark_train_ds.roles
    slama_feats = slama_pipeline.fit_transform(spark_train_ds)
    slama_lf_pds = cast(PandasDataset, slama_feats.to_pandas())

    # now process test part of the data
    slama_test_feats = slama_pipeline.transform(spark_test_ds)
    # dumping resulting datasets
    chkp_train_path = os.path.join(checkpoint_fp_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_train.dump")
    chkp_test_path = os.path.join(checkpoint_fp_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_test.dump")
    dump_data(chkp_train_path, slama_feats[:, slama_pipeline.output_features], cv=cv)
    dump_data(chkp_test_path, slama_test_feats[:, slama_pipeline.output_features], cv=cv)

    # assert sorted(slama_pipeline.output_features) == sorted(lf_pds.features)
    assert sorted(slama_pipeline.output_features) == sorted([f for f in lf_pds.features])
    assert len(set(spark_train_ds.features).difference(slama_feats.features)) == 0
    assert len(set(ds.features).difference(slama_feats.features)) == 0
    assert set(slama_pipeline.output_roles.keys()) == set(f for f in lf_pds.roles.keys())
    assert all([(f in slama_feats.roles) for f in lf_pds.roles.keys()])

    not_equal_roles = [
        feat
        for feat, prole in lf_pds.roles.items()
        if not feat.startswith('nanflg_') and
           not (type(prole) == type(slama_pipeline.output_roles[feat]) == type(slama_feats.roles[feat]))
    ]
    assert len(not_equal_roles) == 0, f"Roles are different: {not_equal_roles}"

    assert set(slama_feats.features) == set(slama_test_feats.features)
    assert slama_feats.roles == slama_test_feats.roles


def compare_mlalgos_by_quality(spark: SparkSession, cv: int, config: Dict[str, Any],
                               fp_lama_clazz, ml_algo_lama_clazz, ml_algo_spark_clazz,
                               pipeline_name: str, ml_alg_kwargs,
                               ml_kwargs_lama: Optional[Dict[str, Any]] = None,
                               ml_kwargs_spark: Optional[Dict[str, Any]] = None):
    checkpoint_dir = '/opt/test_checkpoints/feature_pipelines'
    path = config['path']
    task_name = config['task_type']
    ds_name = os.path.basename(os.path.splitext(path)[0])

    dump_train_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_train.dump") \
        if checkpoint_dir is not None else None
    dump_test_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_test.dump") \
        if checkpoint_dir is not None else None

    train_res = load_dump_if_exist(spark, dump_train_path)
    test_res = load_dump_if_exist(spark, dump_test_path)
    if not train_res or not test_res:
        raise ValueError("Dataset should be processed with feature pipeline "
                         "and the corresponding dump should exist. Please, run corresponding non-quality test first.")
    dumped_train_ds, _ = train_res
    dumped_test_ds, _ = test_res

    if not ml_kwargs_lama:
        ml_kwargs_lama = dict()

    if not ml_kwargs_spark:
        ml_kwargs_spark = dict()

    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    train_pdf = pd.read_csv(config['train_path'], **read_csv_args)
    test_pdf = pd.read_csv(config['test_path'], **read_csv_args)
    # train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=100)
    reader = PandasToPandasReader(task=Task(task_name), cv=cv, advanced_roles=False, random_state=seed)
    train_ds = reader.fit_read(train_pdf, roles=config['roles'])
    test_ds = reader.read(test_pdf, add_array_attrs=True)
    lama_pipeline = fp_lama_clazz(**ml_alg_kwargs)
    lama_feats = lama_pipeline.fit_transform(train_ds)
    lama_test_feats = lama_pipeline.transform(test_ds)
    lama_feats = lama_feats if ml_algo_lama_clazz == BoostLGBM else lama_feats.to_numpy()
    train_valid = FoldsIterator(lama_feats.to_numpy())
    if not USE_FOLDS_VALIDATION:
        train_valid = train_valid.convert_to_holdout_iterator()
    ml_algo = ml_algo_lama_clazz(freeze_defaults=False, **ml_kwargs_lama)
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    assert ml_algo is not None
    test_pred = ml_algo.predict(lama_test_feats)
    # test_pred = ml_algo.predict(test_ds)
    score = train_valid.train.task.get_dataset_metric()
    lama_oof_metric = score(oof_pred)
    lama_test_metric = score(test_pred)

    train_valid = SparkFoldsIterator(dumped_train_ds, n_folds=cv)
    if not USE_FOLDS_VALIDATION:
        train_valid = train_valid.convert_to_holdout_iterator()
    ml_algo = ml_algo_spark_clazz(cacher_key='test', freeze_defaults=False, **ml_kwargs_spark)
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    ml_algo = cast(SparkTabularMLAlgo, ml_algo)
    assert ml_algo is not None
    test_pred = ml_algo.predict(dumped_test_ds)
    score = train_valid.train.task.get_dataset_metric()
    spark_based_oof_metric = score(oof_pred[:, ml_algo.prediction_feature])
    spark_based_test_metric = score(test_pred[:, ml_algo.prediction_feature])

    print(f"LAMA oof: {lama_oof_metric}. Lama test: {lama_test_metric}")
    print(f"Spark oof: {spark_based_oof_metric}. Spark test: {spark_based_test_metric}")

    max_diff_in_percents = 0.05

    assert spark_based_test_metric > lama_test_metric or abs(
        (lama_test_metric - spark_based_test_metric) / max(lama_test_metric,
                                                           spark_based_test_metric)) < max_diff_in_percents
    assert spark_based_test_metric > lama_test_metric or abs(
        (lama_test_metric - spark_based_test_metric) / min(lama_test_metric,
                                                           spark_based_test_metric)) < max_diff_in_percents


@pytest.mark.parametrize("ds_config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_linear_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    compare_feature_pipelines(spark, cv, ds_config, LinearFeatures, SparkLinearFeatures,
                              ml_alg_kwargs, 'linear_features')


@pytest.mark.parametrize("ds_config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_lgbadv_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    compare_feature_pipelines(spark, cv, ds_config, LGBAdvancedPipeline, SparkLGBAdvancedPipeline,
                              ml_alg_kwargs, 'lgbadv_features')


@pytest.mark.parametrize("ds_config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_lgbsimple_features(spark: SparkSession, ds_config: Dict[str, Any], cv: int):
    compare_feature_pipelines(spark, cv, ds_config, LGBSimpleFeatures, SparkLGBSimpleFeatures,
                              dict(), 'lgbsimple_features')


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_quality_linear_features(spark: SparkSession, config: Dict[str, Any], cv: int):
    compare_feature_pipelines_by_quality(spark, cv, config, LinearFeatures, LinearLBFGS,
                                         ml_alg_kwargs, 'linear_features')


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_quality_lgbadv_features(spark: SparkSession, config: Dict[str, Any], cv: int):
    compare_feature_pipelines_by_quality(spark, cv, config, LGBAdvancedPipeline, BoostLGBM,
                                         ml_alg_kwargs, 'lgbadv_features')


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_quality_lgbsimple_features(spark: SparkSession, config: Dict[str, Any], cv: int):
    compare_feature_pipelines_by_quality(spark, cv, config, LGBSimpleFeatures, BoostLGBM,
                                         dict(), 'lgbsimple_features')


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_quality_mlalgo_linearlgbfs(spark: SparkSession, config: Dict[str, Any], cv: int):
    compare_mlalgos_by_quality(spark, cv, config, LinearFeatures, LinearLBFGS, SparkLinearLBFGS, 'linear_features',
                               ml_alg_kwargs)


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_quality_mlalgo_boostlgbm(spark: SparkSession, config: Dict[str, Any], cv: int):
    compare_mlalgos_by_quality(spark, cv, config, LGBAdvancedPipeline, BoostLGBM, SparkBoostLGBM, 'lgbadv_features',
                               ml_alg_kwargs)


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(**DATASETS_ARG)])
def test_quality_mlalgo_simple_features_boostlgbm(spark: SparkSession, config: Dict[str, Any], cv: int):
    compare_mlalgos_by_quality(spark, cv, config, LGBSimpleFeatures, BoostLGBM, SparkBoostLGBM, 'lgbsimple_features',
                               {})
