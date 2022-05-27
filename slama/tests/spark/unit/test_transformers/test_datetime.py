from typing import List

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.spark.transformers.datetime import SparkBaseDiffTransformer, SparkDateSeasonsTransformer, \
    SparkTimeToNumTransformer
from lightautoml.tasks import Task
from lightautoml.transformers.datetime import TimeToNum, BaseDiff, DateSeasons
from .. import DatasetForTest, compare_sparkml_by_content, spark as spark_sess

spark = spark_sess

DATASETS = [

    DatasetForTest(df=pd.DataFrame(data={
        "night": [
            "2000-01-01 00:00:00",
            np.nan,
            "2020-01-01 00:00:00",
            "2025-01-01 00:00:00",
            "2100-01-01 00:00:00",
        ],
        "morning": [
            "2000-01-01 06:00:00",
            "2017-01-01 06:00:00",
            "2020-01-01 06:00:00",
            None,
            "2100-01-01 06:00:00",
        ],
        "day": [
            np.nan,
            "2017-01-01 12:00:00",
            "2020-01-01 12:00:00",
            "2025-01-01 12:00:00",
            "2100-01-01 12:00:00",
        ],
        "evening": [
            "2000-01-01 20:00:00",
            "2017-01-01 20:00:00",
            "2020-01-01 20:00:00",
            "2025-01-01 20:00:00",
            "2100-01-01 20:00:00",
        ],
    }), default_role=DatetimeRole()),

    DatasetForTest(df=pd.DataFrame(data={
        "night": [
            "2000-06-05 00:00:00",
            "2020-01-01 00:00:00",
            "2025-05-01 00:00:00",
            "2100-08-01 00:00:00",
        ],
        "morning": [
            "2000-03-01 06:05:00",
            "2017-02-01 06:05:00",
            "2020-04-01 06:05:00",
            "2100-11-01 06:05:00",
        ],
        "day": [
            "2017-05-01 12:00:30",
            "2020-02-01 12:00:30",
            "2025-01-01 12:00:30",
            "2100-01-01 12:00:30",
        ],
        "evening": [
            "2000-01-01 20:00:00",
            "2020-01-01 20:00:00",
            "2025-01-01 20:00:00",
            "2100-01-01 20:00:00",
        ],
    }), default_role=DatetimeRole(country="Russia", seasonality=('y', 'm', 'd', 'wd', 'hour', 'min', 'sec', 'ms', 'ns'))),

]


@pytest.mark.parametrize("dataset", DATASETS)
def test_time_to_num(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task('binary'))

    compare_sparkml_by_content(spark, ds, TimeToNum(), SparkTimeToNumTransformer(input_cols=ds.features, input_roles=ds.roles))


@pytest.mark.parametrize("dataset", DATASETS)
def test_base_diff(spark: SparkSession, dataset: DatasetForTest):

    columns: List[str] = dataset.dataset.columns
    middle = int(len(columns)/2)
    base_names = columns[:middle]
    diff_names = columns[middle:]

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    compare_sparkml_by_content(
        spark,
        ds,
        BaseDiff(base_names=base_names, diff_names=diff_names),
        SparkBaseDiffTransformer(input_roles=ds.roles, base_names=base_names, diff_names=diff_names)
    )


@pytest.mark.parametrize("dataset", [DATASETS[1]])
def test_date_seasons(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    compare_sparkml_by_content(spark, ds, DateSeasons(),
                               SparkDateSeasonsTransformer(input_cols=ds.features, input_roles=ds.roles))
