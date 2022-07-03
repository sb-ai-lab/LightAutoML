from copy import deepcopy
from typing import Optional, Union, Set

import numpy as np

from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.base import SparkFeaturesPipeline, SparkTabularDataFeatures
from lightautoml.spark.transformers.base import SparkChangeRolesTransformer, SparkUnionTransformer, \
    SparkSequentialTransformer, SparkEstOrTrans
from lightautoml.spark.transformers.categorical import SparkOrdinalEncoderEstimator
from lightautoml.spark.transformers.datetime import SparkTimeToNumTransformer


class SparkLGBSimpleFeatures(SparkFeaturesPipeline, SparkTabularDataFeatures):
    """Creates simple pipeline for tree based models.

    Simple but is ok for select features.
    Numeric stay as is, Datetime transforms to numeric.
    Categorical label encoding.
    Maps input to output features exactly one-to-one.

    """
    def __init__(self, cacher_key: str = 'default_cacher'):
        super().__init__(cacher_key)

    def _get_input_features(self) -> Set[str]:
        return set(self.input_features)

    def create_pipeline(self, train: SparkDataset) -> Union[SparkUnionTransformer, SparkSequentialTransformer]:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """
        transformers_list = []

        # process categories
        categories = self._cols_by_role(train, "Category")
        if len(categories) > 0:
            roles = {f: train.roles[f] for f in categories}
            cat_processing = SparkOrdinalEncoderEstimator(input_cols=categories,
                                                          input_roles=roles,
                                                          subs=None,
                                                          random_state=42)
            transformers_list.append(cat_processing)

        # process datetimes
        datetimes = self._cols_by_role(train, "Datetime")
        if len(datetimes) > 0:
            roles = {f: train.roles[f] for f in datetimes}
            dt_processing = SparkTimeToNumTransformer(input_cols=datetimes,
                                                      input_roles=roles)
            transformers_list.append(dt_processing)

        transformers_list.append(self.get_numeric_data(train))

        union_all = SparkUnionTransformer([x for x in transformers_list if x is not None])

        return union_all


class SparkLGBAdvancedPipeline(SparkFeaturesPipeline, SparkTabularDataFeatures):
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
            cacher_key: str = 'default_cacher',
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
            cacher_key=cacher_key,
            multiclass_te_co=multiclass_te_co,
            top_intersections=top_intersections,
            max_intersection_depth=max_intersection_depth,
            subsample=subsample,
            feats_imp=feats_imp,
            auto_unique_co=auto_unique_co,
            output_categories=output_categories,
            ascending_by_cardinality=False,
        )

    def _get_input_features(self) -> Set[str]:
        return set(self.input_features)

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Transformer.

        """

        features = train.features
        roles = deepcopy(train.roles)
        transformer_list = []
        
        target_encoder = self.get_target_encoder(train)

        output_category_role = (
            CategoryRole(np.float32, label_encoded=True) if self.output_categories else NumericRole(np.float32)
        )

        # handle categorical feats
        # split categories by handling type. This pipe use 3 encodings - freq/label/target/ordinal
        # 1 - separate freqs. It does not need label encoding
        stage = self.get_freq_encoding(train)
        transformer_list.append(stage)

        # 2 - check different target encoding parts and split (ohe is the same as auto - no ohe in gbm)
        auto = self._cols_by_role(train, "Category", encoding_type="auto") \
               + self._cols_by_role(train, "Category", encoding_type="ohe")

        if self.output_categories:
            le = (
                    auto
                    + self._cols_by_role(train, "Category", encoding_type="oof")
                    + self._cols_by_role(train, "Category", encoding_type="int")
            )
            te = []
            ordinal = None

        else:
            le = self._cols_by_role(train, "Category", encoding_type="int")
            ordinal = self._cols_by_role(train, "Category", ordinal=True)

            if target_encoder is not None:
                te = self._cols_by_role(train, "Category", encoding_type="oof")
                # split auto categories by unique values cnt
                un_values = self.get_uniques_cnt(train, auto)
                te = te + [x for x in un_values.index if un_values[x] > self.auto_unique_co]
                ordinal = ordinal + list(set(auto) - set(te))

            else:
                te = []
                ordinal = ordinal + auto + self._cols_by_role(train, "Category", encoding_type="oof")

            ordinal = sorted(list(set(ordinal)))

        # get label encoded categories
        le_part = self.get_categorical_raw(train, le)
        if le_part is not None:
            # le_part = SequentialTransformer([le_part, ChangeRoles(output_category_role)])
            change_roles_stage = SparkChangeRolesTransformer(input_cols=le_part.getOutputCols(),
                                                             input_roles=le_part.getOutputRoles(),
                                                             role=output_category_role)
            le_part = SparkSequentialTransformer([le_part, change_roles_stage])
            transformer_list.append(le_part)

        # get target encoded part
        te_part = self.get_categorical_raw(train, te)
        if te_part is not None:
            target_encoder_stage = target_encoder(
                input_cols=te_part.getOutputCols(),
                input_roles=te_part.getOutputRoles(),
                task_name=train.task.name,
                folds_column=train.folds_column,
                target_column=train.target_column,
                do_replace_columns=True
            )
            te_part = SparkSequentialTransformer([te_part, target_encoder_stage])
            transformer_list.append(te_part)

        # get intersection of top categories
        intersections = self.get_categorical_intersections(train)
        if intersections is not None:
            if target_encoder is not None:
                target_encoder_stage = target_encoder(
                    input_cols=intersections.getOutputCols(),
                    input_roles=intersections.getOutputRoles(),
                    task_name=train.task.name,
                    folds_column=train.folds_column,
                    target_column=train.target_column,
                    do_replace_columns=True
                )
                ints_part = SparkSequentialTransformer([intersections, target_encoder_stage])
            else:
                change_roles_stage = SparkChangeRolesTransformer(
                    input_cols=intersections.getOutputCols(),
                    input_roles=intersections.getOutputRoles(),
                    role=output_category_role
                )
                ints_part = SparkSequentialTransformer([intersections, change_roles_stage])

            transformer_list.append(ints_part)

        transformer_list.append(self.get_numeric_data(train))
        transformer_list.append(self.get_ordinal_encoding(train, ordinal))
        transformer_list.append(self.get_datetime_diffs(train))
        transformer_list.append(self.get_datetime_seasons(train, NumericRole(np.float32)))

        # final pipeline
        union_all = SparkUnionTransformer([x for x in transformer_list if x is not None])

        return union_all
