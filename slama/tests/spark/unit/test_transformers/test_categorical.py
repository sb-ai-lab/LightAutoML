from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator, SparkFreqEncoderEstimator, \
    SparkOrdinalEncoderEstimator, SparkCatIntersectionsEstimator, SparkTargetEncoderEstimator, \
    SparkMulticlassTargetEncoderEstimator
from lightautoml.tasks import Task
from lightautoml.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder, CatIntersectstions, \
    TargetEncoder, MultiClassTargetEncoder
from .. import DatasetForTest, compare_sparkml_by_content, spark as spark_sess, compare_sparkml_by_metadata
from ..dataset_utils import get_test_datasets

spark = spark_sess

CV = 5

DATASETS = [

    # DatasetForTest("unit/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("tests/spark/unit/resources/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage", "WoodDeckSF"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32),
                       "WoodDeckSF": CategoryRole(bool)
                   })


    # DatasetForTest("unit/resources/datasets/house_prices.csv",
    #                columns=["Id", "MSZoning", "WoodDeckSF"],
    #                roles={
    #                    "Id": CategoryRole(np.int32),
    #                    "MSZoning": CategoryRole(str),
    #                    "WoodDeckSF": CategoryRole(bool)
    #                })
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_sparkml_label_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    transformer = SparkLabelEncoderEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    compare_sparkml_by_metadata(spark, ds, LabelEncoder(), transformer, compare_feature_distributions=True)


@pytest.mark.parametrize("dataset", DATASETS)
def test_freq_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    transformer = SparkFreqEncoderEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    compare_sparkml_by_content(spark, ds, FreqEncoder(), transformer)


@pytest.mark.parametrize("dataset", DATASETS)
def test_ordinal_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    transformer = SparkOrdinalEncoderEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    compare_sparkml_by_content(spark, ds, OrdinalEncoder(), transformer)


@pytest.mark.parametrize("dataset", DATASETS)
def test_cat_intersections(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    # read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    # pdf = pd.read_csv(config['path'], **read_csv_args)
    #
    # reader = PandasToPandasReader(task=Task(config["task_type"]), cv=CV, advanced_roles=False)
    # train_ds = reader.fit_read(pdf, roles=config['roles'])
    #
    # # ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    # le_cols = get_columns_by_role(train_ds, "Category")
    # train_ds = train_ds[:, le_cols]
    #
    transformer = SparkCatIntersectionsEstimator(
        input_cols=ds.features,
        input_roles=ds.roles
    )
    #
    compare_sparkml_by_metadata(spark, ds, CatIntersectstions(), transformer, compare_feature_distributions=True)
    # compare_sparkml_by_content(spark, ds, CatIntersectstions(), transformer)


@pytest.mark.parametrize("dataset", DATASETS)
def test_target_encoder(spark: SparkSession, dataset: DatasetForTest):
    # reader = PandasToPandasReader(task=Task("binary"), cv=CV, advanced_roles=False)
    # train_ds = reader.fit_read(dataset.dataset, roles=dataset.roles)

    target = pd.Series(np.random.choice(a=[0, 1], size=dataset.dataset.shape[0], p=[0.5, 0.5]))
    folds = pd.Series(np.random.choice(a=[i for i in range(CV)],
                                       size=dataset.dataset.shape[0], p=[1.0 / CV for i in range(CV)]))

    train_ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"), target=target, folds=folds)

    le = LabelEncoder()
    train_ds = le.fit_transform(train_ds)
    train_ds = train_ds.to_pandas()

    transformer = SparkTargetEncoderEstimator(
        input_cols=train_ds.features,
        input_roles=train_ds.roles,
        task_name=train_ds.task.name,
        target_column='target',
        folds_column='folds'
    )

    compare_sparkml_by_metadata(spark, train_ds, TargetEncoder(), transformer, compare_feature_distributions=True)


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(dataset="used_cars_dataset")])
def test_target_encoder_real_datasets(spark: SparkSession, config: Dict[str, Any], cv: int):
    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    pdf = pd.read_csv(config['path'], **read_csv_args)

    reader = PandasToPandasReader(task=Task(config["task_type"]), cv=CV, advanced_roles=False)
    train_ds = reader.fit_read(pdf, roles=config['roles'])

    le_cols = get_columns_by_role(train_ds, "Category")
    train_ds = train_ds[:, le_cols]

    le = LabelEncoder()
    train_ds = le.fit_transform(train_ds)
    train_ds = train_ds.to_pandas()

    transformer = SparkTargetEncoderEstimator(
        input_cols=train_ds.features,
        input_roles=train_ds.roles,
        task_name=train_ds.task.name,
        target_column='target',
        folds_column='folds'
    )

    compare_sparkml_by_metadata(spark, train_ds, TargetEncoder(), transformer, compare_feature_distributions=True)


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(dataset='ipums_97')])
def test_multi_target_encoder(spark: SparkSession, config: Dict[str, Any], cv: int):
    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    pdf = pd.read_csv(config['path'], **read_csv_args)

    reader = PandasToPandasReader(task=Task(config["task_type"]), cv=CV, advanced_roles=False)
    train_ds = reader.fit_read(pdf, roles=config['roles'])

    le_cols = get_columns_by_role(train_ds, "Category")
    train_ds = train_ds[:, le_cols]

    le = LabelEncoder()
    train_ds = le.fit_transform(train_ds)
    train_ds = train_ds.to_pandas()

    transformer = SparkMulticlassTargetEncoderEstimator(
        input_cols=train_ds.features,
        input_roles=train_ds.roles,
        task_name=train_ds.task.name,
        target_column='target',
        folds_column='folds'
    )

    compare_sparkml_by_metadata(spark, train_ds, MultiClassTargetEncoder(), transformer, compare_feature_distributions=True)
