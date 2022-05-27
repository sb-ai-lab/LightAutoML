import logging
import pickle
import random
from typing import cast

import pandas as pd
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession

from lightautoml.automl.blend import BestModelSelector, Blender, WeightedBlender
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.automl.blend import SparkBestModelSelector, SparkWeightedBlender

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM as SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.pipelines.ml.nested_ml_pipe import SparkNestedTabularMLPipeline as SparkNestedTabularMLPipeline
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.transformers.base import ColumnsSelectorTransformer, DropColumnsTransformer
from lightautoml.spark.utils import log_exec_time, VERBOSE_LOGGING_FORMAT
from lightautoml.spark.validation.iterators import SparkDummyIterator
from lightautoml.tasks import Task
from .. import from_pandas_to_spark, spark as spark_sess, compare_obtained_datasets

import numpy as np
import pyspark.sql.functions as F

from ..test_auto_ml.utils import DummySparkMLPipeline, DummyMLAlgo

spark = spark_sess


logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def test_weighted_blender(spark: SparkSession):
    target_col = "some_target"
    folds_col = "folds"
    n_classes = 10
    models_count = 4

    data = [
        {
            SparkDataset.ID_COLUMN: i,
            "a": i, "b": 100 + i, "c": 100 * i,
            target_col: random.randint(0, n_classes),
            folds_col: random.randint(0, 2)
        }
        for i in range(100)
    ]

    roles = {"a": NumericRole(), "b": NumericRole(), "c": NumericRole()}

    data_sdf = spark.createDataFrame(data)
    data_sds = SparkDataset(data=data_sdf, task=SparkTask("multiclass"),
                            roles=roles, target=target_col, folds=folds_col)

    pipes = [
        SparkMLPipeline(cacher_key='test_cache', ml_algos=[DummyMLAlgo(n_classes, name=f"dummy_0_{i}")])
        for i in range(models_count)
    ]

    for pipe in pipes:
        data_sds = pipe.fit_predict(SparkDummyIterator(data_sds))

    preds_roles = {c: role for c, role in data_sds.roles.items() if c not in roles}

    sdf = data_sds.data.drop(*list(roles.keys())).cache()
    sdf.write.mode('overwrite').format('noop').save()
    ml_ds = data_sds.empty()
    ml_ds.set_data(sdf, list(preds_roles.keys()), preds_roles)

    swb = SparkWeightedBlender(max_iters=1, max_inner_iters=1)
    with log_exec_time('Blender fit_predict'):
        blended_sds, filtered_pipes = swb.fit_predict(ml_ds, pipes)
        blended_sds.data.write.mode('overwrite').format('noop').save()

    with log_exec_time('Blender predict'):
        transformed_preds_sdf = swb.transformer.transform(ml_ds.data)
        transformed_preds_sdf.write.mode('overwrite').format('noop').save()

    assert len(swb.output_roles) == 1
    prediction, role = list(swb.output_roles.items())[0]
    if data_sds.task.name in ["binary", "multiclass"]:
        assert isinstance(role, NumericVectorOrArrayRole)
    else:
        assert isinstance(role, NumericRole)
    assert prediction in blended_sds.data.columns
    assert prediction in blended_sds.roles
    assert blended_sds.roles[prediction] == role
    assert prediction in transformed_preds_sdf.columns

