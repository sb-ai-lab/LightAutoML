"""Iterative feature selector."""

import logging

from copy import deepcopy
from typing import Optional, cast, Iterator

import numpy as np

from pandas import Series

from pyspark.sql import functions as F
from pyspark.sql.functions import shuffle
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StructField

from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset, SparkDataset
from ...ml_algo.base import MLAlgo
from lightautoml.spark.pipelines.selection.base import SparkImportanceEstimator

import pandas as pd

logger = logging.getLogger(__name__)


class SparkNpPermutationImportanceEstimator(SparkImportanceEstimator):
    """Permutation importance based estimator.

    Importance calculate, using random permutation
    of items in single column for each feature.

    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: seed for random generation of features permutation.

        """
        super().__init__()
        self.random_state = random_state

    def fit(
        self,
        train_valid: Optional[TrainValidIterator] = None,
        ml_algo: Optional[MLAlgo] = None,
        preds: Optional[LAMLDataset] = None,
    ):
        """Find importances for each feature in dataset.

        Args:
            train_valid: Initial dataset iterator.
            ml_algo: Algorithm.
            preds: Predicted target values for validation dataset.

        """
        logger.info(f"Starting importance estimating with {type(self)}")

        normal_score = ml_algo.score(preds)
        logger.debug(f"Normal score = {normal_score}")

        valid_data = cast(SparkDataset, train_valid.get_validation_data())

        permutation_importance = {}

        for it, feat in enumerate(valid_data.features):
            logger.info(f"Start processing ({it},{feat})")
            df = valid_data.data

            field: StructField = df.schema[feat]

            @pandas_udf(returnType=field.dataType)
            def permutate(arrs: Iterator[pd.Series]) -> Iterator[pd.Series]:
                permutator = np.random.RandomState(seed=self.random_state)
                # one may get list of arrs and concatenate them to perform permutation
                # in the whole partition
                for x in arrs:
                    px = permutator.permutation(x)
                    yield pd.Series(px)

            permutated_df = df.withColumn(feat, permutate(feat))

            ds: SparkDataset = valid_data.empty()
            ds.set_data(
                permutated_df,
                valid_data.features,
                valid_data.roles
            )
            logger.debug("Dataframe with shuffled column prepared")

            # Calculate predict and metric
            new_preds = ml_algo.predict(ds)
            shuffled_score = ml_algo.score(new_preds)
            logger.debug(
                "Shuffled score for col {} = {}, difference with normal = {}".format(
                    feat, shuffled_score, normal_score - shuffled_score
                )
            )
            permutation_importance[feat] = normal_score - shuffled_score

        self.raw_importances = Series(permutation_importance).sort_values(ascending=False)

        logger.info(f"Finished importance estimating with {type(self)}")
