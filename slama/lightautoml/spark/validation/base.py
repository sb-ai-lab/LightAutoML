import functools
from abc import ABC
from copy import copy
from typing import Tuple, cast, Optional, List, Sequence

from pyspark.sql import functions as F

from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.reader.base import RolesDict
from lightautoml.spark import VALIDATION_COLUMN
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.utils import SparkDataFrame
from lightautoml.spark.pipelines.base import InputFeaturesAndRoles
from lightautoml.spark.pipelines.features.base import SparkFeaturesPipeline
from lightautoml.validation.base import TrainValidIterator


class SparkBaseTrainValidIterator(TrainValidIterator, InputFeaturesAndRoles, ABC):
    """
    Implements applying selection pipeline and feature pipeline to SparkDataset.
    """

    TRAIN_VAL_COLUMN = VALIDATION_COLUMN

    def __init__(self, train: SparkDataset, input_roles: Optional[RolesDict] = None):
        assert train.folds_column in train.data.columns
        super().__init__(train)
        if not input_roles:
            input_roles = train.roles
        self._input_roles = input_roles

    def __next__(self) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        """Define how to get next object.

        Returns:
            a tuple with:
            - full dataset (both train and valid parts with column containing the bool feature),
            - train part of the dataset
            - validation part of the dataset.

        """
        ...

    @property
    def features(self) -> List[str]:
        return self.input_features

    def apply_selector(self, selector: SelectionPipeline) -> "SparkBaseTrainValidIterator":
        """Select features on train data.

        Check if selector is fitted.
        If not - fit and then perform selection.
        If fitted, check if it's ok to apply.

        Args:
            selector: Uses for feature selection.

        Returns:
            Dataset with selected features.

        """
        sel_train_valid = copy(self)
        sel_train_valid.train = self.train[:, list(self.input_roles.keys())]

        if not selector.is_fitted:
            selector.fit(sel_train_valid)
            sfp = cast(SparkFeaturesPipeline, selector.features_pipeline)
            if sfp is not None:
                sfp.release_cache()

        train_valid = copy(self)
        # we don't need to create transformer for subselecting
        # because train_valid.input_roles is used in fit_... methods
        # of features pipelines and ml_algo to define columns they work with
        train_valid.input_roles = {feat: self.input_roles[feat]
                                   for feat in selector.selected_features}

        return train_valid

    def apply_feature_pipeline(self, features_pipeline: SparkFeaturesPipeline) -> "SparkBaseTrainValidIterator":
        features_pipeline.input_roles = self.input_roles
        train_valid = cast(SparkBaseTrainValidIterator, super().apply_feature_pipeline(features_pipeline))
        train_valid.input_roles = features_pipeline.output_roles
        return train_valid

    def combine_val_preds(self, val_preds: Sequence[SparkDataFrame], include_train: bool = False) -> SparkDataFrame:
        # depending on train_valid logic there may be several ways of treating predictions results:
        # 1. for folds iterators - just union the results, it will yield the full train dataset
        # 2. for holdout iterators - create None predictions in train_part and union with valid part
        # 3. for custom iterators which may put the same records in
        #   different folds: union + groupby + (optionally) union with None-fied train_part
        # 4. for dummy - do nothing
        raise NotImplementedError()

    def _split_by_fold(self, fold: int) -> Tuple[SparkDataset, SparkDataset, SparkDataset]:
        train = cast(SparkDataset, self.train)
        is_val_col = (
            F.when(F.col(self.train.folds_column) != fold, F.lit(0))
            .otherwise(F.lit(1))
            .alias(self.TRAIN_VAL_COLUMN)
        )

        sdf = train.data.select('*', is_val_col)
        train_part_sdf = sdf.where(F.col(self.TRAIN_VAL_COLUMN) == 0).drop(self.TRAIN_VAL_COLUMN)
        valid_part_sdf = sdf.where(F.col(self.TRAIN_VAL_COLUMN) == 1).drop(self.TRAIN_VAL_COLUMN)

        train_ds = cast(SparkDataset, self.train.empty())
        train_ds.set_data(sdf, self.train.features, self.train.roles)

        train_part_ds = cast(SparkDataset, self.train.empty())
        train_part_ds.set_data(train_part_sdf, self.train.features, self.train.roles)

        valid_part_ds = cast(SparkDataset, self.train.empty())
        valid_part_ds.set_data(valid_part_sdf, self.train.features, self.train.roles)

        return train_ds, train_part_ds, valid_part_ds
