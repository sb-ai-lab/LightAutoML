"""Pipeline for tree based models."""

from typing import Optional
from typing import Union

import numpy as np

from ...dataset.np_pd_dataset import NumpyDataset
from ...dataset.np_pd_dataset import PandasDataset
from ...dataset.roles import CategoryRole
from ...dataset.roles import NumericRole
from ...transformers.base import ChangeRoles
from ...transformers.base import ColumnsSelector
from ...transformers.base import ConvertDataset
from ...transformers.base import LAMLTransformer
from ...transformers.base import SequentialTransformer
from ...transformers.base import UnionTransformer
from ...transformers.categorical import OrdinalEncoder
from ...transformers.datetime import TimeToNum
from ..selection.base import ImportanceEstimator
from ..utils import get_columns_by_role
from .base import FeaturesPipeline
from .base import TabularDataFeatures


NumpyOrPandas = Union[PandasDataset, NumpyDataset]


class LGBSimpleFeatures(FeaturesPipeline):
    """Creates simple pipeline for tree based models.

    Simple but is ok for select features.
    Numeric stay as is, Datetime transforms to numeric.
    Categorical label encoding.
    Maps input to output features exactly one-to-one.

    """

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """
        # TODO: Transformer params to config
        transformers_list = []

        # process categories
        categories = get_columns_by_role(train, "Category")
        if len(categories) > 0:
            cat_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=categories),
                    OrdinalEncoder(subs=None, random_state=42),
                    # ChangeRoles(NumericRole(np.float32))
                ]
            )
            transformers_list.append(cat_processing)

        # process datetimes
        datetimes = get_columns_by_role(train, "Datetime")
        if len(datetimes) > 0:
            dt_processing = SequentialTransformer([ColumnsSelector(keys=datetimes), TimeToNum()])
            transformers_list.append(dt_processing)

        # process numbers
        numerics = get_columns_by_role(train, "Numeric")
        if len(numerics) > 0:
            num_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=numerics),
                    ConvertDataset(dataset_type=NumpyDataset),
                ]
            )
            transformers_list.append(num_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class LGBAdvancedPipeline(FeaturesPipeline, TabularDataFeatures):
    """Create advanced pipeline for trees based models.

    Includes:

        - Different cats and numbers handling according to role params.
        - Dates handling - extracting seasons and create datediffs.
        - Create categorical intersections.

    """

    def __init__(
        self,
        feats_imp: Optional[ImportanceEstimator] = None,
        top_intersections: int = 5,
        max_intersection_depth: int = 3,
        subsample: Optional[Union[int, float]] = None,
        multiclass_te_co: int = 3,
        auto_unique_co: int = 10,
        output_categories: bool = False,
        **kwargs
    ):
        """

        Args:
            feats_imp: Features importances mapping.
            top_intersections: Max number of categories
              to generate intersections.
            max_intersection_depth: Max depth of cat intersection.
            subsample: Subsample to calc data statistics.
            multiclass_te_co: Cutoff if use target encoding in cat
              handling on multiclass task if number of classes is high.
            auto_unique_co: Switch to target encoding if high cardinality.

        """
        super().__init__(
            multiclass_te_co=multiclass_te_co,
            top_intersections=top_intersections,
            max_intersection_depth=max_intersection_depth,
            subsample=subsample,
            feats_imp=feats_imp,
            auto_unique_co=auto_unique_co,
            output_categories=output_categories,
            ascending_by_cardinality=False,
        )

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """

        transformer_list = []
        target_encoder = self.get_target_encoder(train)

        output_category_role = (
            CategoryRole(np.float32, label_encoded=True) if self.output_categories else NumericRole(np.float32)
        )

        # handle categorical feats
        # split categories by handling type. This pipe use 3 encodings - freq/label/target/ordinal
        # 1 - separate freqs. It does not need label encoding
        transformer_list.append(self.get_freq_encoding(train))

        # 2 - check different target encoding parts and split (ohe is the same as auto - no ohe in gbm)
        auto = get_columns_by_role(train, "Category", encoding_type="auto") + get_columns_by_role(
            train, "Category", encoding_type="ohe"
        )

        if self.output_categories:
            le = (
                auto
                + get_columns_by_role(train, "Category", encoding_type="oof")
                + get_columns_by_role(train, "Category", encoding_type="int")
            )
            te = []
            ordinal = None

        else:
            le = get_columns_by_role(train, "Category", encoding_type="int")
            ordinal = get_columns_by_role(train, "Category", ordinal=True)

            if target_encoder is not None:
                te = get_columns_by_role(train, "Category", encoding_type="oof")
                # split auto categories by unique values cnt
                un_values = self.get_uniques_cnt(train, auto)
                te = te + [x for x in un_values.index if un_values[x] > self.auto_unique_co]
                ordinal = ordinal + list(set(auto) - set(te))

            else:
                te = []
                ordinal = ordinal + auto + get_columns_by_role(train, "Category", encoding_type="oof")

            ordinal = sorted(list(set(ordinal)))

        # get label encoded categories
        le_part = self.get_categorical_raw(train, le)
        if le_part is not None:
            le_part = SequentialTransformer([le_part, ChangeRoles(output_category_role)])
            transformer_list.append(le_part)

        # get target encoded part
        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            te_part = SequentialTransformer([te_part, target_encoder()])
            transformer_list.append(te_part)

        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                ints_part = SequentialTransformer([intersections, target_encoder()])
            else:
                ints_part = SequentialTransformer([intersections, ChangeRoles(output_category_role)])

            transformer_list.append(ints_part)

        # add numeric pipeline
        transformer_list.append(self.get_numeric_data(train))
        transformer_list.append(self.get_ordinal_encoding(train, ordinal))
        # add difference with base date
        transformer_list.append(self.get_datetime_diffs(train))
        # add datetime seasonality
        transformer_list.append(self.get_datetime_seasons(train, NumericRole(np.float32)))

        # final pipeline
        union_all = UnionTransformer([x for x in transformer_list if x is not None])

        return union_all
