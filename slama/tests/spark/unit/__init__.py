import time
from copy import copy
from typing import Tuple, get_args, cast, List, Optional, Dict, Union

import numpy as np
import pandas as pd
import pytest
from pyspark.ml import Estimator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.dataset.roles import ColumnRole, CategoryRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.transformers.base import ObsoleteSparkTransformer, SparkBaseEstimator, SparkBaseTransformer, \
    SparkColumnsAndRoles
from lightautoml.spark.utils import log_exec_time
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.numeric import NumpyTransformable


# NOTE!!!
# All tests require PYSPARK_PYTHON env variable to be set
# for example: PYSPARK_PYTHON=/home/nikolay/.conda/envs/LAMA/bin/python
JAR_PATH = 'jars/spark-lightautoml_2.12-0.1.jar'


@pytest.fixture(scope="session")
def spark() -> SparkSession:

    spark = (
        SparkSession
        .builder
        .appName("LAMA-test-app")
        .master("local[4]")
        .config("spark.driver.memory", "8g")
        .config("spark.jars", JAR_PATH)
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.sql.shuffle.partitions", 200)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


@pytest.fixture(scope="session")
def spark_with_deps() -> SparkSession:
    spark = SparkSession.builder.appName("LAMA-test-app")\
        .master("local[1]") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


def compare_feature_distrs_in_datasets(lama_df, spark_df, diff_proc=0.05):
    stats_names = ['count', 'mean',
                   'std', 'min',
                   '25%', '50%',
                   '75%', 'max']

    lama_df_stats = lama_df.describe()
    spark_df_stats = spark_df.describe()
    columns = list(lama_df_stats)

    found_difference = False

    for col in columns:
        if col not in list(spark_df):
            print(col)
            spark_df = spark_df.rename(columns={'oof__'+col:col})
            spark_df_stats = spark_df.describe()

        lama_col_uniques = lama_df[col].unique()
        spark_col_uniques = spark_df[col].unique()
        lama_col_uniques_num = len(lama_col_uniques)
        spark_col_uniques_num = len(spark_col_uniques)
        # comparing uniques:\n",
        if abs(lama_col_uniques_num - spark_col_uniques_num) > lama_col_uniques_num * diff_proc:
            found_difference = True
            print()
            print(f'Difference between uniques {lama_col_uniques_num} (lama) and {spark_col_uniques_num} (spark)')
            print('Lama: ', lama_col_uniques)
            print('Spark: ', spark_col_uniques)
            print()
        for stats_col in stats_names:
            if abs(lama_df_stats[col][stats_col] - spark_df_stats[col][stats_col]) > abs(lama_df_stats[col][stats_col] * diff_proc):
                found_difference = True
                print(f'Difference in col {col} and stats {stats_col} between {lama_df_stats[col][stats_col]} (lama) and {spark_df_stats[col][stats_col]} (spark)')

    assert not found_difference


def compare_datasets(ds: PandasDataset,
                     transformed_ds: LAMLDataset,
                     transformed_sds: SparkDataset,
                     compare_feature_distributions: bool = True,
                     compare_content: bool = False):
    lama_np_ds = cast(NumpyTransformable, transformed_ds).to_numpy()
    spark_np_ds = transformed_sds.to_pandas()
    print(f"\nTransformed SPRK: \n{spark_np_ds.data[spark_np_ds.features]}")
    # for row in spark_np_ds:
    #     print(row)

    # One can compare lists, sets and dicts in Python using '==' operator
    # For dicts, for instance, pythons checks presence of the same keya in both dicts
    # and then compare values with the same keys in both dicts using __eq__ operator of the entities
    # https://hg.python.org/cpython/file/6f535c725b27/Objects/dictobject.c#l1839
    # https://docs.pytest.org/en/6.2.x/example/reportingdemo.html#tbreportdemo

    initial_features = ds.features
    sfeatures = set(spark_np_ds.features)

    assert all((f in sfeatures) for f in initial_features), \
        f"Not all initial features are presented in Spark features " \
        f"LAMA initial features: {sorted(initial_features)}" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    assert ds.roles == {f: spark_np_ds.roles[f] for f in initial_features}, "Initial roles are not equal"

    # compare shapes
    assert lama_np_ds.shape[0] == spark_np_ds.shape[0], "Shapes are not equals"
    # including initial features
    assert lama_np_ds.shape[1] + ds.shape[1] == spark_np_ds.shape[1], "Shapes are not equals"

    spark_np_ds = spark_np_ds[:, lama_np_ds.features].to_numpy()
    # compare independent of feature ordering
    assert all((f in sfeatures) for f in lama_np_ds.features), \
        f"Not all LAMA features are presented in Spark features\n" \
        f"LAMA: {sorted(lama_np_ds.features)}\n" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    # compare roles equality for the columns
    assert lama_np_ds.roles == {f: spark_np_ds.roles[f] for f in lama_np_ds.features}, "Roles are not equal"

    if compare_feature_distributions:
        trans_data: pd.DataFrame = lama_np_ds.to_pandas().data
        trans_data_result: pd.DataFrame = spark_np_ds.to_pandas().data
        compare_feature_distrs_in_datasets(trans_data[lama_np_ds.features], trans_data_result[lama_np_ds.features])

    if compare_content:
        # features: List[int] = [i for i, _ in sorted(enumerate(transformed_ds.features), key=lambda x: x[1])]
        feat_map = {f: i for i, f in enumerate(spark_np_ds.features)}
        features: List[int] = [feat_map[f] for f in lama_np_ds.features]

        trans_data: np.ndarray = lama_np_ds.data
        trans_data_result: np.ndarray = spark_np_ds.data

        # compare content equality of numpy arrays
        assert np.allclose(trans_data[:, features], trans_data_result[:, features], equal_nan=True), \
            f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
            f"\n\nLAMA: \n{trans_data}" \
            f"\n\nSpark: \n{trans_data_result}"


def compare_sparkml_transformers_results(spark: SparkSession,
                                         ds: PandasDataset,
                                         t_lama: LAMLTransformer,
                                         t_spark: Union[SparkBaseEstimator, SparkBaseTransformer],
                                         compare_feature_distributions: bool = True,
                                         compare_content: bool = True):
    """
    Args:
        spark: session to be used for calculating the example
        ds: a dataset to be transformered by LAMA and Spark transformers
        t_lama: LAMA's version of the transformer
        t_spark: spark's version of the transformer
        compare_metadata_only: if True comapre only metadata of the resulting pair of datasets - columns
        count and their labels (e.g. features), roles and shapez

    Returns:
        A tuple of (LAMA transformed dataset, Spark transformed dataset)
    """
    sds = from_pandas_to_spark(ds, spark, ds.target)
    transformed_ds = t_lama.fit_transform(ds)

    assert isinstance(transformed_ds, get_args(NumpyTransformable)), \
        f"The returned dataset doesn't belong numpy covertable types {NumpyTransformable} and " \
        f"thus cannot be checked againt the resulting spark dataset." \
        f"The dataset's type is {type(transformed_ds)}"

    with log_exec_time("SPARK EXEC"):
        if isinstance(t_spark, Estimator):
            t_spark = t_spark.fit(sds.data)

    transformed_df = t_spark.transform(sds.data)
    transformed_sds = SparkColumnsAndRoles.make_dataset(t_spark, sds, transformed_df)

    compare_datasets(ds, transformed_ds, transformed_sds, compare_feature_distributions, compare_content)

    # now compare dataset after simple transformation
    transformed_ds = t_lama.transform(ds)
    transformed_df = t_spark.transform(sds.data)
    transformed_sds = SparkColumnsAndRoles.make_dataset(t_spark, sds, transformed_df)

    compare_datasets(ds, transformed_ds, transformed_sds, compare_feature_distributions, compare_content)


def compare_sparkml_by_content(spark: SparkSession,
                       ds: PandasDataset,
                       t_lama: LAMLTransformer,
                       t_spark: Union[SparkBaseEstimator, SparkBaseTransformer]):
    """
        Args:
            spark: session to be used for calculating the example
            ds: a dataset to be transformered by LAMA and Spark transformers
            t_lama: LAMA's version of the transformer
            t_spark: spark's version of the transformer

        Returns:
            A tuple of (LAMA transformed dataset, Spark transformed dataset)
        """
    compare_sparkml_transformers_results(spark, ds, t_lama, t_spark)


def compare_sparkml_by_metadata(spark: SparkSession,
                                ds: PandasDataset,
                                t_lama: LAMLTransformer,
                                t_spark: Union[SparkBaseEstimator, SparkBaseTransformer],
                                compare_feature_distributions: bool = False) -> Tuple[NumpyDataset, NumpyDataset]:
    """
        Args:
            spark: session to be used for calculating the example
            ds: a dataset to be transformered by LAMA and Spark transformers
            t_lama: LAMA's version of the transformer
            t_spark: spark's version of the transformer

        Returns:
            A tuple of (LAMA transformed dataset, Spark transformed dataset)
        """
    compare_sparkml_transformers_results(spark, ds, t_lama, t_spark,
                                         compare_feature_distributions=compare_feature_distributions,
                                         compare_content=False)


def compare_transformers_results(spark: SparkSession,
                                 ds: PandasDataset,
                                 t_lama: LAMLTransformer,
                                 t_spark: ObsoleteSparkTransformer,
                                 compare_metadata_only: bool = False) -> Tuple[NumpyDataset, NumpyDataset]:
    """
    Args:
        spark: session to be used for calculating the example
        ds: a dataset to be transformered by LAMA and Spark transformers
        t_lama: LAMA's version of the transformer
        t_spark: spark's version of the transformer
        compare_metadata_only: if True comapre only metadata of the resulting pair of datasets - columns
        count and their labels (e.g. features), roles and shapez

    Returns:
        A tuple of (LAMA transformed dataset, Spark transformed dataset)
    """
    sds = from_pandas_to_spark(ds, spark, ds.target)

    t_lama.fit(ds)
    transformed_ds = t_lama.transform(ds)

    # print(f"Transformed LAMA: {transformed_ds.data}")

    assert isinstance(transformed_ds, get_args(NumpyTransformable)), \
        f"The returned dataset doesn't belong numpy covertable types {NumpyTransformable} and " \
        f"thus cannot be checked againt the resulting spark dataset." \
        f"The dataset's type is {type(transformed_ds)}"

    lama_np_ds = cast(NumpyTransformable, transformed_ds).to_numpy()

    print(f"\nTransformed LAMA: \n{lama_np_ds}")
    # for row in lama_np_ds:
    #     print(row)

    with log_exec_time():
        t_spark.fit(sds)
        transformed_sds = t_spark.transform(sds)

    spark_np_ds = transformed_sds.to_numpy()
    print(f"\nTransformed SPRK: \n{spark_np_ds}")
    # for row in spark_np_ds:
    #     print(row)

    # One can compare lists, sets and dicts in Python using '==' operator
    # For dicts, for instance, pythons checks presence of the same keya in both dicts
    # and then compare values with the same keys in both dicts using __eq__ operator of the entities
    # https://hg.python.org/cpython/file/6f535c725b27/Objects/dictobject.c#l1839
    # https://docs.pytest.org/en/6.2.x/example/reportingdemo.html#tbreportdemo

    # compare independent of feature ordering
    assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
        f"List of features are not equal\n" \
        f"LAMA: {sorted(lama_np_ds.features)}\n" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    # compare roles equality for the columns
    assert lama_np_ds.roles == spark_np_ds.roles, "Roles are not equal"

    # compare shapes
    assert lama_np_ds.shape == spark_np_ds.shape, "Shapes are not equals"

    if not compare_metadata_only:
        features: List[int] = [i for i, _ in sorted(enumerate(transformed_ds.features), key=lambda x: x[1])]

        trans_data: np.ndarray = lama_np_ds.data
        trans_data_result: np.ndarray = spark_np_ds.data

        # compare content equality of numpy arrays
        assert np.allclose(trans_data[:, features], trans_data_result[:, features], equal_nan=True), \
            f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
            f"\n\nLAMA: \n{trans_data}" \
            f"\n\nSpark: \n{trans_data_result}"

    return lama_np_ds, spark_np_ds


def compare_by_content(spark: SparkSession,
                       ds: PandasDataset,
                       t_lama: LAMLTransformer,
                       t_spark: ObsoleteSparkTransformer) -> Tuple[NumpyDataset, NumpyDataset]:
    """
        Args:
            spark: session to be used for calculating the example
            ds: a dataset to be transformered by LAMA and Spark transformers
            t_lama: LAMA's version of the transformer
            t_spark: spark's version of the transformer

        Returns:
            A tuple of (LAMA transformed dataset, Spark transformed dataset)
        """
    return compare_transformers_results(spark, ds, t_lama, t_spark, compare_metadata_only=False)


def compare_by_metadata(spark: SparkSession,
                        ds: PandasDataset,
                        t_lama: LAMLTransformer,
                        t_spark: ObsoleteSparkTransformer) -> Tuple[NumpyDataset, NumpyDataset]:
    """

        Args:
            spark: session to be used for calculating the example
            ds: a dataset to be transformered by LAMA and Spark transformers
            t_lama: LAMA's version of the transformer
            t_spark: spark's version of the transformer

        Returns:
            A tuple of (LAMA transformed dataset, Spark transformed dataset)

        NOTE: Content of the datasets WON'T be checked for equality.
        This function should be used only to compare stochastic-based transformers
    """
    return compare_transformers_results(spark, ds, t_lama, t_spark, compare_metadata_only=True)


def smoke_check(spark: SparkSession, ds: PandasDataset, t_spark: ObsoleteSparkTransformer) -> NumpyDataset:
    sds = from_pandas_to_spark(ds, spark)

    t_spark.fit(sds)
    transformed_sds = t_spark.transform(sds)

    spark_np_ds = transformed_sds.to_numpy()

    return spark_np_ds


class DatasetForTest:
    def __init__(self, path: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 columns: Optional[List[str]] = None,
                 roles: Optional[Dict] = None,
                 default_role: Optional[ColumnRole] = None):

        if path is not None:
            self.dataset = pd.read_csv(path)
        else:
            self.dataset = df

        if columns is not None:
            self.dataset = self.dataset[columns]

        if roles is None:
            self.roles = {name: default_role for name in self.dataset.columns}
        else:
            self.roles = roles


def from_pandas_to_spark(p: PandasDataset,
                         spark: SparkSession,
                         target: Optional[pd.Series] = None,
                         folds: Optional[pd.Series] = None,
                         task: Optional[SparkTask] = None,
                         to_vector: bool = False,
                         fill_folds_with_zeros_if_not_present: bool = False) -> SparkDataset:
    pdf = cast(pd.DataFrame, p.data)
    pdf = pdf.copy()
    pdf[SparkDataset.ID_COLUMN] = pdf.index

    roles = copy(p.roles)

    kwargs = dict()

    if target is not None:
        pdf['target'] = target
        kwargs['target'] = 'target'

    if 'target' in p.__dict__ and p.target is not None:
        pdf['target'] = p.target
        kwargs['target'] = 'target'

    if folds is not None:
        pdf['folds'] = folds
        kwargs['folds'] = 'folds'

    if 'folds' in p.__dict__ and p.folds is not None:
        pdf['folds'] = p.folds
        kwargs['folds'] = 'folds'

    obj_columns = list(pdf.select_dtypes(include=['object']))
    pdf[obj_columns] = pdf[obj_columns].astype(str)
    sdf = spark.createDataFrame(data=pdf)

    if to_vector:
        cols = [c for c in pdf.columns if c != SparkDataset.ID_COLUMN]
        general_feat = cols[0]
        sdf = sdf.select(SparkDataset.ID_COLUMN, F.array(*cols).alias(general_feat))
        roles = {general_feat: NumericVectorOrArrayRole(len(cols), f"{general_feat}_{{}}", dtype=roles[cols[0]].dtype)}

    if task:
        spark_task = task
    elif p.task:
        spark_task = SparkTask(p.task.name)
    else:
        spark_task = None

    return SparkDataset(sdf, roles=roles, task=spark_task, **kwargs)


def compare_obtained_datasets(lama_ds: NumpyDataset, spark_ds: SparkDataset):
    lama_np_ds = cast(NumpyTransformable, lama_ds).to_numpy()
    spark_np_ds = spark_ds.to_numpy()

    assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
        f"List of features are not equal\n" \
        f"LAMA: {sorted(lama_np_ds.features)}\n" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    # compare roles equality for the columns
    assert lama_np_ds.roles == spark_np_ds.roles, \
        f"Roles are not equal.\n" \
        f"LAMA: {lama_np_ds.roles}\n" \
        f"Spark: {spark_np_ds.roles}"

    # compare shapes
    assert lama_np_ds.shape == spark_np_ds.shape

    lama_data: np.ndarray = lama_np_ds.data
    spark_data: np.ndarray = spark_np_ds.data
    # features: List[int] = [i for i, _ in sorted(enumerate(lama_np_ds.features), key=lambda x: x[1])]

    # assert np.allclose(
    #     np.sort(lama_data[:, features], axis=0), np.sort(spark_data[:, features], axis=0),
    #     equal_nan=True
    # ), \
    #     f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
    #     f"\n\nLAMA: \n{lama_data}" \
    #     f"\n\nSpark: \n{spark_data}"

    lama_feature_column_ids = {feature: i for i, feature in sorted(enumerate(lama_np_ds.features), key=lambda x: x[1]) }
    spark_feature_column_ids = {feature: i for i, feature in sorted(enumerate(spark_np_ds.features), key=lambda x: x[1]) }
    for feature in lama_feature_column_ids.keys():
        lama_column_id = [lama_feature_column_ids[feature]]
        spark_column_id = [spark_feature_column_ids[feature]]
        result = np.allclose(
            np.sort(lama_data[:, lama_column_id], axis=0), np.sort(spark_data[:, spark_column_id], axis=0),
            equal_nan=True
        )
        # print(f"feature: {feature}, result: {result}")
        assert result, \
            f"Results of the LAMA's transformer column '{feature}' and the Spark based transformer column '{feature}' are not equal: " \
            f"\n\nLAMA: \n{lama_data[:, lama_column_id]}" \
            f"\n\nSpark: \n{spark_data[:, spark_column_id]}"
