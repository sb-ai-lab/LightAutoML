"""Linear models features."""

from typing import Optional
from typing import Union

import numpy as np

from ...dataset.np_pd_dataset import NumpyDataset
from ...dataset.np_pd_dataset import PandasDataset
from ...dataset.roles import CategoryRole
from ...transformers.base import ChangeRoles
from ...transformers.base import LAMLTransformer
from ...transformers.base import SequentialTransformer
from ...transformers.base import UnionTransformer
from ...transformers.categorical import LabelEncoder
from ...transformers.categorical import OHEEncoder
from ...transformers.numeric import FillInf
from ...transformers.numeric import FillnaMedian
from ...transformers.numeric import LogOdds
from ...transformers.numeric import NaNFlags
from ...transformers.numeric import StandardScaler
from ..selection.base import ImportanceEstimator
from ..utils import get_columns_by_role
from .base import FeaturesPipeline
from .base import TabularDataFeatures


NumpyOrPandas = Union[PandasDataset, NumpyDataset]


class LinearFeatures(FeaturesPipeline, TabularDataFeatures):
    """
    Creates pipeline for linear models and nnets.

    Includes:

        - Create categorical intersections.
        - OHE or embed idx encoding for categories.
        - Other cats to numbers ways if defined in role params.
        - Standartization and nan handling for numbers.
        - Numbers discretization if needed.
        - Dates handling.
        - Handling probs (output of lower level models).

    """

    def __init__(
        self,
        feats_imp: Optional[ImportanceEstimator] = None,
        top_intersections: int = 5,
        max_bin_count: int = 10,
        max_intersection_depth: int = 3,
        subsample: Optional[Union[int, float]] = None,
        sparse_ohe: Union[str, bool] = "auto",
        auto_unique_co: int = 50,
        output_categories: bool = True,
        multiclass_te_co: int = 3,
        **kwargs
    ):
        """

        Args:
            feats_imp: Features importances mapping.
            top_intersections: Max number of categories
              to generate intersections.
            max_bin_count: Max number of bins to discretize numbers.
            max_intersection_depth: Max depth of cat intersection.
            subsample: Subsample to calc data statistics.
            sparse_ohe: Should we output sparse if ohe encoding
              was used during cat handling.
            auto_unique_co: Switch to target encoding if high cardinality.
            output_categories: Output encoded categories or embed idxs.
            multiclass_te_co: Cutoff if use target encoding in cat handling
              on multiclass task if number of classes is high.

        """
        assert max_bin_count is None or max_bin_count > 1, "Max bin count should be >= 2 or None"

        super().__init__(
            multiclass_te=False,
            top_intersections=top_intersections,
            max_intersection_depth=max_intersection_depth,
            subsample=subsample,
            feats_imp=feats_imp,
            auto_unique_co=auto_unique_co,
            output_categories=output_categories,
            ascending_by_cardinality=True,
            max_bin_count=max_bin_count,
            sparse_ohe=sparse_ohe,
            multiclass_te_co=multiclass_te_co,
        )

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """Create linear pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """
        transformers_list = []
        dense_list = []
        sparse_list = []
        probs_list = []
        target_encoder = self.get_target_encoder(train)
        te_list = dense_list if train.task.name == "reg" else probs_list

        # handle categorical feats
        # split categories by handling type. This pipe use 4 encodings - freq/label/target/ohe/ordinal
        # 1 - separate freqs. It does not need label encoding
        dense_list.append(self.get_freq_encoding(train))

        # 2 - check 'auto' type (int is the same - no label encoded numbers in linear models)
        auto = get_columns_by_role(train, "Category", encoding_type="auto") + get_columns_by_role(
            train, "Category", encoding_type="int"
        )

        # if self.output_categories or target_encoder is None:
        if target_encoder is None:
            le = (
                auto
                + get_columns_by_role(train, "Category", encoding_type="oof")
                + get_columns_by_role(train, "Category", encoding_type="ohe")
            )
            te = []

        else:
            te = get_columns_by_role(train, "Category", encoding_type="oof")
            le = get_columns_by_role(train, "Category", encoding_type="ohe")
            # split auto categories by unique values cnt
            un_values = self.get_uniques_cnt(train, auto)
            te = te + [x for x in un_values.index if un_values[x] > self.auto_unique_co]
            le = le + list(set(auto) - set(te))

        # get label encoded categories
        sparse_list.append(self.get_categorical_raw(train, le))

        # get target encoded categories
        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            te_part = SequentialTransformer([te_part, target_encoder()])
            te_list.append(te_part)

        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                ints_part = SequentialTransformer([intersections, target_encoder()])
                te_list.append(ints_part)
            else:
                sparse_list.append(intersections)

        # add datetime seasonality
        seas_cats = self.get_datetime_seasons(train, CategoryRole(np.int32))
        if seas_cats is not None:
            sparse_list.append(SequentialTransformer([seas_cats, LabelEncoder()]))

        # get quantile binning
        sparse_list.append(self.get_binned_data(train))
        # add numeric pipeline wo probs
        dense_list.append(self.get_numeric_data(train, prob=False))
        # add ordinal categories
        dense_list.append(self.get_ordinal_encoding(train))
        # add probs
        probs_list.append(self.get_numeric_data(train, prob=True))
        # add difference with base date
        dense_list.append(self.get_datetime_diffs(train))

        # combine it all together
        # handle probs if exists
        probs_list = [x for x in probs_list if x is not None]
        if len(probs_list) > 0:
            probs_pipe = UnionTransformer(probs_list)
            probs_pipe = SequentialTransformer([probs_pipe, LogOdds()])
            dense_list.append(probs_pipe)

        # handle dense
        dense_list = [x for x in dense_list if x is not None]
        if len(dense_list) > 0:
            # standartize, fillna, add null flags
            dense_pipe = SequentialTransformer(
                [
                    UnionTransformer(dense_list),
                    UnionTransformer(
                        [
                            SequentialTransformer([FillInf(), FillnaMedian(), StandardScaler()]),
                            NaNFlags(),
                        ]
                    ),
                ]
            )
            transformers_list.append(dense_pipe)

        # handle categories - cast to float32 if categories are inputs or make ohe
        sparse_list = [x for x in sparse_list if x is not None]
        if len(sparse_list) > 0:
            sparse_pipe = UnionTransformer(sparse_list)
            if self.output_categories:
                final = ChangeRoles(CategoryRole(np.float32))
            else:
                if self.sparse_ohe == "auto":
                    final = OHEEncoder(total_feats_cnt=train.shape[1])
                else:
                    final = OHEEncoder(make_sparse=self.sparse_ohe)
            sparse_pipe = SequentialTransformer([sparse_pipe, final])

            transformers_list.append(sparse_pipe)

        # final pipeline
        union_all = UnionTransformer(transformers_list)

        return union_all
