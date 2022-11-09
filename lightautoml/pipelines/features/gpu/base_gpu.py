"""Basic classes for features generation (GPU version)."""

from copy import copy
from typing import List, Optional, Union

import dask_cudf
import numpy as np
import pandas as pd

from lightautoml.pipelines.features.base import TabularDataFeatures
from lightautoml.pipelines.utils import get_columns_by_role

try:
    from lightautoml.dataset.gpu.gpu_dataset import (
        CudfDataset,
        CupyDataset,
        DaskCudfDataset,
    )
except:
    pass
from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.transformers.base import (
    ChangeRoles,
    ColumnsSelector,
    ConvertDataset,
    LAMLTransformer,
    SequentialTransformer,
)

try:
    from lightautoml.transformers.gpu.categorical_gpu import (
        CatIntersections_gpu,
        FreqEncoder_gpu,
        LabelEncoder_gpu,
        MultiClassTargetEncoder_gpu,
        OrdinalEncoder_gpu,
        TargetEncoder_gpu,
        MultioutputTargetEncoder_gpu
    )
    from lightautoml.transformers.gpu.datetime_gpu import BaseDiff_gpu, DateSeasons_gpu
    from lightautoml.transformers.gpu.numeric_gpu import QuantileBinning_gpu
except:
    pass

GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class TabularDataFeatures_gpu(TabularDataFeatures):
    """Helper class contains basic features transformations for tabular data.

    This method can de shared by all tabular feature pipelines,
    to simplify ``.create_automl`` definition.
    """

    def get_datetime_diffs(self, train: GpuDataset) -> Optional[LAMLTransformer]:
        """Difference for all datetimes with base date.

        Args:
            train: Dataset with train data.

        Returns:
            Transformer or ``None`` if no required features.

        """
        base_dates, datetimes = self.get_cols_for_datetime(train)
        if len(datetimes) == 0 or len(base_dates) == 0:
            return

        dt_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=list(set(datetimes + base_dates))),
                BaseDiff_gpu(base_names=base_dates, diff_names=datetimes),
            ]
        )
        return dt_processing

    def get_datetime_seasons(
        self, train: GpuDataset, outp_role: Optional[ColumnRole] = None
    ) -> Optional[LAMLTransformer]:
        """Get season params from dates.

        Args:
            train: Dataset with train data.
            outp_role: Role associated with output features.

        Returns:
            Transformer or ``None`` if no required features.

        """
        _, datetimes = self.get_cols_for_datetime(train)
        for col in copy(datetimes):
            if (
                len(train.roles[col].seasonality) == 0
                and train.roles[col].country is None
            ):
                datetimes.remove(col)

        if len(datetimes) == 0:
            return

        if outp_role is None:
            outp_role = NumericRole(np.float32)

        date_as_cat = SequentialTransformer(
            [ColumnsSelector(keys=datetimes), DateSeasons_gpu(outp_role)]
        )
        return date_as_cat

    @staticmethod
    def get_numeric_data(
        train: GpuDataset,
        feats_to_select: Optional[List[str]] = None,
        prob: Optional[bool] = None,
    ) -> Optional[LAMLTransformer]:
        """Select numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.
            prob: Probability flag.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            if prob is None:
                feats_to_select = get_columns_by_role(train, "Numeric")
            else:
                feats_to_select = get_columns_by_role(train, "Numeric", prob=prob)

        if len(feats_to_select) == 0:
            return

        dataset_type = type(train)

        num_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                ConvertDataset(dataset_type=dataset_type),
                ChangeRoles(NumericRole(np.float32)),
            ]
        )

        return num_processing

    @staticmethod
    def get_freq_encoding(
        train: GpuDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[LAMLTransformer]:
        """Get frequency encoding part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(
                train, "Category", encoding_type="freq"
            )

        if len(feats_to_select) == 0:
            return

        cat_processing = SequentialTransformer(
            [ColumnsSelector(keys=feats_to_select), FreqEncoder_gpu()]
        )
        return cat_processing

    def get_ordinal_encoding(
        self, train: GpuDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[LAMLTransformer]:
        """Get order encoded part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, "Category", ordinal=True)

        if len(feats_to_select) == 0:
            return

        cat_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                OrdinalEncoder_gpu(subs=self.subsample, random_state=self.random_state),
            ]
        )
        return cat_processing

    def get_categorical_raw(
        self, train: GpuDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[LAMLTransformer]:
        """Get label encoded categories data.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """

        if feats_to_select is None:
            feats_to_select = []
            for i in ["auto", "oof", "int", "ohe"]:
                feats_to_select.extend(
                    get_columns_by_role(train, "Category", encoding_type=i)
                )

        if len(feats_to_select) == 0:
            return

        cat_processing = [
            ColumnsSelector(keys=feats_to_select),
            LabelEncoder_gpu(subs=self.subsample, random_state=self.random_state),
        ]
        cat_processing = SequentialTransformer(cat_processing)
        return cat_processing

    def get_target_encoder(self, train: GpuDataset) -> Optional[type]:
        """Get target encoder func for dataset.

        Args:
            train: Dataset with train data.

        Returns:
            Class

        """
        target_encoder = None
        if train.folds is not None:
            if train.task.name in ["binary", "reg"]:
                target_encoder = TargetEncoder_gpu
            elif (train.task.name == "multi:reg") or (train.task.name == "multilabel"):
                n_classes = train.target.shape
                if type(train) == DaskCudfDataset:
                    n_classes = n_classes.compute()[1]
                else:
                    n_classes = n_classes[1]
                if n_classes <= self.multiclass_te_co:
                    target_encoder = MultioutputTargetEncoder_gpu
            else:
                target_max = train.target.max()
                if type(train) == DaskCudfDataset:
                    target_max = target_max.compute()
                n_classes = target_max + 1

                if n_classes <= self.multiclass_te_co:
                    target_encoder = MultiClassTargetEncoder_gpu

        return target_encoder

    def get_binned_data(
        self, train: GpuDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[LAMLTransformer]:
        """Get encoded quantiles of numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: features to hanlde. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = get_columns_by_role(train, "Numeric", discretization=True)

        if len(feats_to_select) == 0:
            return

        binned_processing = SequentialTransformer(
            [
                ColumnsSelector(keys=feats_to_select),
                QuantileBinning_gpu(nbins=self.max_bin_count),
            ]
        )
        return binned_processing

    def get_categorical_intersections(
        self, train: GpuDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[LAMLTransformer]:
        """Get transformer that implements categorical intersections.

        Args:
            train: Dataset with train data.
            feats_to_select: features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """

        if feats_to_select is None:

            categories = get_columns_by_role(train, "Category")
            feats_to_select = categories

            if len(categories) <= 1:
                return

            elif len(categories) > self.top_intersections:
                feats_to_select = self.get_top_categories(train, self.top_intersections)

        elif len(feats_to_select) <= 1:
            return

        cat_processing = [
            ColumnsSelector(keys=feats_to_select),
            CatIntersections_gpu(
                subs=self.subsample,
                random_state=self.random_state,
                max_depth=self.max_intersection_depth,
            ),
        ]
        cat_processing = SequentialTransformer(cat_processing)

        return cat_processing

    def get_uniques_cnt(self, train: GpuDataset, feats: List[str]) -> pd.Series:
        """Get unique values cnt.

        Args:
            train: Dataset with train data.
            feats: Features names.

        Returns:
            Series.

        """
        un = []
        if len(feats) > 0:
            data = train.data[feats]

            if self.subsample is not None and self.subsample < len(feats):
                data = data.sample(
                    n=int(self.subsample) if self.subsample > 1 else None,
                    frac=self.subsample if self.subsample <= 1 else None,
                    random_state=self.random_state,
                )

            desc = data.astype(object).describe(include="all")
            un = desc.loc["unique"]
            if type(data) == dask_cudf.DataFrame:
                un = un.compute().astype("int").values[0]
            else:
                un = un.astype("int").values[0].get()
            # can we just transpose dataframe?

        return pd.Series(un, index=feats, dtype="int")
