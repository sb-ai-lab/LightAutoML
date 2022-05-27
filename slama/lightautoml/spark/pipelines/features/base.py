"""Basic classes for features generation."""
import itertools
import logging
from copy import copy
from dataclasses import dataclass
from typing import Any, Callable, cast, Set, Union, Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import toposort
from pandas import DataFrame
from pandas import Series
from pyspark.ml import Transformer, Estimator, Pipeline, PipelineModel
from pyspark.ml.param import Param, Params
from pyspark.sql import functions as F

from lightautoml.dataset.base import RolesDict, LAMLDataset
from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.base import InputFeaturesAndRoles, OutputFeaturesAndRoles
from lightautoml.spark.transformers.base import SparkChangeRolesTransformer, ColumnsSelectorTransformer, \
    DropColumnsTransformer
from lightautoml.spark.transformers.base import SparkBaseEstimator, SparkBaseTransformer, SparkUnionTransformer, \
    SparkSequentialTransformer, SparkEstOrTrans, SparkColumnsAndRoles
from lightautoml.spark.transformers.categorical import SparkCatIntersectionsEstimator, \
    SparkFreqEncoderEstimator, \
    SparkLabelEncoderEstimator, SparkOrdinalEncoderEstimator, SparkMulticlassTargetEncoderEstimator
from lightautoml.spark.transformers.categorical import SparkTargetEncoderEstimator
from lightautoml.spark.transformers.datetime import SparkBaseDiffTransformer, SparkDateSeasonsTransformer
from lightautoml.spark.transformers.numeric import SparkQuantileBinningEstimator
from lightautoml.spark.utils import NoOpTransformer, Cacher, EmptyCacher, warn_if_not_cached, SparkDataFrame

logger = logging.getLogger(__name__)


def build_graph(begin: SparkEstOrTrans):
    """Fill dict that represents graph of estimators and transformers

    Args:
        begin (SparkEstOrTrans): pipeline to extract graph of estimators and transformers
    """
    graph = dict()

    def find_start_end(tr: SparkEstOrTrans) -> Tuple[List[SparkEstOrTrans], List[SparkEstOrTrans]]:
        if isinstance(tr, SparkSequentialTransformer):
            se = [st_or_end for el in tr.transformers for st_or_end in find_start_end(el)]

            starts = se[0]
            ends = se[-1]
            middle = se[1:-1]

            i = 0
            while i < len(middle):
                for new_st, new_end in itertools.product(middle[i], middle[i + 1]):
                    if new_end not in graph:
                        graph[new_end] = set()
                    graph[new_end].add(new_st)
                i += 2

            return starts, ends

        elif isinstance(tr, SparkUnionTransformer):
            se = [find_start_end(el) for el in tr.transformers]
            starts = [s_el for s, _ in se for s_el in s]
            ends = [e_el for _, e in se for e_el in e]
            return starts, ends
        else:
            return [tr], [tr]

    init_starts, final_ends = find_start_end(begin)

    for st in init_starts:
        if st not in graph:
            graph[st] = set()

    return graph


@dataclass
class FittedPipe:
    sdf: SparkDataFrame
    transformer: Transformer
    roles: RolesDict


class SelectTransformer(Transformer):
    """
    Transformer that returns ``pyspark.sql.DataFrame`` with selected columns.
    """

    colsToSelect = Param(Params._dummy(), "colsToSelect",
                        "columns to select from the dataframe")

    def __init__(self, cols_to_select: List[str]):
        """
        Args:
            cols_to_select (List[str]): List of columns to select from input dataframe
        """
        super().__init__()
        self.set(self.colsToSelect, cols_to_select)

    def getColsToSelect(self) -> List[str]:
        return self.getOrDefault(self.colsToSelect)

    def _transform(self, dataset):
        return dataset.select(self.getColsToSelect())


class SparkFeaturesPipeline(InputFeaturesAndRoles, OutputFeaturesAndRoles, FeaturesPipeline):
    """Abstract class.

    Analyze train dataset and create composite transformer
    based on subset of features.
    Instance can be interpreted like Transformer
    (look for :class:`~lightautoml.transformers.base.LAMLTransformer`)
    with delayed initialization (based on dataset metadata)
    Main method, user should define in custom pipeline is ``.create_pipeline``.
    For example, look at
    :class:`~lightautoml.pipelines.features.lgb_pipeline.LGBSimpleFeatures`.
    After FeaturePipeline instance is created, it is used like transformer
    with ``.fit_transform`` and ``.transform`` method.

    """

    def __init__(self, cacher_key: str = 'default_cacher', **kwargs):
        super().__init__(**kwargs)
        self._cacher_key = cacher_key
        self.pipes: List[Callable[[SparkDataset], SparkEstOrTrans]] = [self.create_pipeline]
        self._transformer: Optional[Transformer] = None

    @property
    def transformer(self) -> Optional[Transformer]:
        return self._transformer

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
        """Analyse dataset and create composite transformer.

        Args:
            train: Dataset with train data.

        Returns:
            Composite transformer (pipeline).

        """
        raise NotImplementedError

    def fit_transform(self, train: SparkDataset) -> SparkDataset:
        """Create pipeline and then fit on train data and then transform.

        Args:
            train: Dataset with train data.n

        Returns:
            Dataset with new features.

        """
        logger.info("SparkFeaturePipeline is started")

        assert self.input_features is not None, "Input features should be provided before the fit_transform"
        assert self.input_roles is not None, "Input roles should be provided before the fit_transform"

        fitted_pipe = self._merge_pipes(train)
        self._transformer = fitted_pipe.transformer
        self._output_roles = fitted_pipe.roles

        features = train.features + self.output_features
        roles = copy(train.roles)
        roles.update(self._output_roles)
        transformed_ds = train.empty()
        transformed_ds.set_data(fitted_pipe.sdf, features, roles)

        logger.info("SparkFeaturePipeline is finished")

        return transformed_ds

    def transform(self, test: LAMLDataset) -> LAMLDataset:
        sdf = self._transformer.transform(test.data)

        roles = copy(test.roles)
        roles.update(self.output_roles)

        transformed_ds = test.empty()
        transformed_ds.set_data(sdf, self.output_features, roles)

        return transformed_ds

    def append(self, pipeline):
        if isinstance(pipeline, SparkFeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in pipeline:
            self.pipes.extend(_pipeline.pipes)

        return self

    def prepend(self, pipeline):
        if isinstance(pipeline, SparkFeaturesPipeline):
            pipeline = [pipeline]

        for _pipeline in reversed(pipeline):
            self.pipes = _pipeline.pipes + self.pipes

        return self

    def pop(self, i: int = -1) -> Optional[Callable[[SparkDataset], Estimator]]:
        if len(self.pipes) > 1:
            return self.pipes.pop(i)

    def _merge_pipes(self, data: SparkDataset) -> FittedPipe:
        fitted_pipes = []
        current_sdf = data.data
        for pipe in self.pipes:
            fp = self._optimize_and_fit(current_sdf, pipe(data))
            current_sdf = fp.sdf
            fitted_pipes.append(fp)

        pipeline = PipelineModel(stages=[fp.transformer for fp in fitted_pipes])
        out_roles = dict()
        for fp in fitted_pipes:
            out_roles.update(fp.roles)

        return FittedPipe(sdf=current_sdf, transformer=pipeline, roles=out_roles)

    def _optimize_and_fit(self, train: SparkDataFrame, pipeline: SparkEstOrTrans)\
            -> FittedPipe:
        graph = build_graph(pipeline)
        tr_layers = list(toposort.toposort(graph))

        logger.info(f"Number of layers in the current feature pipeline {self}: {len(tr_layers)}")

        fp_input_features = set(self.input_features)

        current_train: SparkDataFrame = train
        stages = []
        fp_output_cols: List[str] = []
        fp_output_roles: RolesDict = dict()
        for i, layer in enumerate(tr_layers):
            logger.debug(f"Calculating layer ({i + 1}/{len(tr_layers)}). The size of layer: {len(layer)}")
            cols_to_remove = []
            output_cols = []
            layer_model = Pipeline(stages=layer).fit(current_train)
            for j, tr in enumerate(layer):
                logger.debug(f"Processing output columns for transformer ({j + 1}/{len(layer)}): {tr}")
                tr = cast(SparkColumnsAndRoles, tr)
                if tr.getDoReplaceColumns():
                    # ChangeRoles, for instance, may return columns with the same name
                    # thus we don't want to remove these columns
                    self_out_cols = set(tr.getOutputCols())
                    cols_to_remove.extend([f for f in tr.getInputCols() if f not in self_out_cols])
                output_cols.extend(tr.getOutputCols())
                fp_output_roles.update(tr.getOutputRoles())
            fp_output_cols = [c for c in fp_output_cols if c not in cols_to_remove]
            fp_output_cols.extend(output_cols)

            # we cannot really remove input features thus we leave them in the dataframe
            # but they won't in features and roles
            cols_to_remove = set(c for c in cols_to_remove if c not in fp_input_features)

            cacher = Cacher(self._cacher_key)
            pipe = Pipeline(stages=[layer_model, DropColumnsTransformer(list(cols_to_remove)), cacher])
            stages.append(pipe.fit(current_train))
            current_train = cacher.dataset

        fp_output_roles = {f: fp_output_roles[f] for f in fp_output_cols}

        return FittedPipe(current_train, PipelineModel(stages=stages), roles=fp_output_roles)

    def release_cache(self):
        Cacher.release_cache_by_key(self._cacher_key)


class SparkTabularDataFeatures:
    """Helper class contains basic features transformations for tabular data.

    This method can de shared by all tabular feature pipelines,
    to simplify ``.create_automl`` definition.
    """

    def __init__(self, **kwargs: Any):
        """Set default parameters for tabular pipeline constructor.

        Args:
            **kwargs: Additional parameters.

        """
        self.multiclass_te_co = 3
        self.top_intersections = 5
        self.max_intersection_depth = 3
        self.subsample = 0.1 #10000
        self.random_state = 42
        self.feats_imp = None
        self.ascending_by_cardinality = False

        self.max_bin_count = 10
        self.sparse_ohe = "auto"

        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    def _get_input_features(self) -> Set[str]:
        raise NotImplementedError()

    def _cols_by_role(self, dataset: SparkDataset, role_name: str, **kwargs: Any) -> List[str]:
        cols = get_columns_by_role(dataset, role_name, **kwargs)
        filtered_cols = [col for col in cols if col in self._get_input_features()]
        return filtered_cols

    def get_cols_for_datetime(self, train: SparkDataset) -> Tuple[List[str], List[str]]:
        """Get datetime columns to calculate features.

        Args:
            train: Dataset with train data.

        Returns:
            2 list of features names - base dates and common dates.

        """
        base_dates = self._cols_by_role(train, "Datetime", base_date=True)
        datetimes = self._cols_by_role(train, "Datetime", base_date=False) + self._cols_by_role(
            train, "Datetime", base_date=True, base_feats=True
        )

        return base_dates, datetimes

    def get_datetime_diffs(self, train: SparkDataset) -> Optional[SparkBaseTransformer]:
        """Difference for all datetimes with base date.

        Args:
            train: Dataset with train data.

        Returns:
            Transformer or ``None`` if no required features.

        """
        base_dates, datetimes = self.get_cols_for_datetime(train)
        if len(datetimes) == 0 or len(base_dates) == 0:
            return None

        roles = {f: train.roles[f] for f in itertools.chain(base_dates, datetimes)}

        base_diff = SparkBaseDiffTransformer(
            input_roles=roles,
            base_names=base_dates,
            diff_names=datetimes
        )

        return base_diff

    def get_datetime_seasons(
        self, train: SparkDataset, outp_role: Optional[ColumnRole] = None
    ) -> Optional[SparkBaseTransformer]:
        """Get season params from dates.

        Args:
            train: Dataset with train data.
            outp_role: Role associated with output features.

        Returns:
            Transformer or ``None`` if no required features.

        """
        _, datetimes = self.get_cols_for_datetime(train)
        for col in copy(datetimes):
            if len(train.roles[col].seasonality) == 0 and train.roles[col].country is None:
                datetimes.remove(col)

        if len(datetimes) == 0:
            return

        if outp_role is None:
            outp_role = NumericRole(np.float32)

        roles = {f: train.roles[f] for f in datetimes}

        date_as_cat = SparkDateSeasonsTransformer(input_cols=datetimes, input_roles=roles, output_role=outp_role)

        return date_as_cat

    def get_numeric_data(
        self,
        train: SparkDataset,
        feats_to_select: Optional[List[str]] = None,
        prob: Optional[bool] = None,
    ) -> Optional[SparkBaseTransformer]:
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
                feats_to_select = self._cols_by_role(train, "Numeric")
            else:
                feats_to_select = self._cols_by_role(train, "Numeric", prob=prob)

        if len(feats_to_select) == 0:
            return None

        roles = {f: train.roles[f] for f in feats_to_select}

        num_processing = SparkChangeRolesTransformer(input_cols=feats_to_select,
                                                     input_roles=roles,
                                                     role=NumericRole(np.float32))

        return num_processing

    def get_freq_encoding(self, train: SparkDataset, feats_to_select: Optional[List[str]] = None) \
            -> Optional[SparkBaseEstimator]:
        """Get frequency encoding part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Category", encoding_type="freq")

        if len(feats_to_select) == 0:
            return None

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkFreqEncoderEstimator(input_cols=feats_to_select, input_roles=roles)

        return cat_processing

    def get_ordinal_encoding(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get order encoded part.

        Args:
            train: Dataset with train data.
            feats_to_select: Features to handle. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Category", ordinal=True)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        ord = SparkOrdinalEncoderEstimator(input_cols=feats_to_select,
                                           input_roles=roles,
                                           subs=self.subsample,
                                           random_state=self.random_state)

        return ord

    def get_categorical_raw(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
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
                feats = self._cols_by_role(train, "Category", encoding_type=i)
                feats_to_select.extend(feats)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkLabelEncoderEstimator(input_cols=feats_to_select,
                                                    input_roles=roles,
                                                    subs=self.subsample,
                                                    random_state=self.random_state)
        return cat_processing

    def get_target_encoder(self, train: SparkDataset) -> Optional[type]:
        """Get target encoder func for dataset.

        Args:
            train: Dataset with train data.

        Returns:
            Class

        """
        target_encoder = None
        if train.folds is not None:
            if train.task.name in ["binary", "reg"]:
                target_encoder = SparkTargetEncoderEstimator
            else:
                result = train.data.select(F.max(train.target_column).alias("max")).first()
                n_classes = result['max'] + 1

                if n_classes <= self.multiclass_te_co:
                    target_encoder = SparkMulticlassTargetEncoderEstimator

        return target_encoder

    def get_binned_data(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
        """Get encoded quantiles of numeric features.

        Args:
            train: Dataset with train data.
            feats_to_select: features to hanlde. If ``None`` - default filter.

        Returns:
            Transformer.

        """
        if feats_to_select is None:
            feats_to_select = self._cols_by_role(train, "Numeric", discretization=True)

        if len(feats_to_select) == 0:
            return

        roles = {f: train.roles[f] for f in feats_to_select}

        binned_processing = SparkQuantileBinningEstimator(
            input_cols=feats_to_select,
            input_roles=roles,
            nbins=self.max_bin_count
        )

        return binned_processing

    def get_categorical_intersections(
        self, train: SparkDataset, feats_to_select: Optional[List[str]] = None
    ) -> Optional[SparkBaseEstimator]:
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

        roles = {f: train.roles[f] for f in feats_to_select}

        cat_processing = SparkCatIntersectionsEstimator(input_cols=feats_to_select,
                                                        input_roles=roles,
                                                        max_depth=self.max_intersection_depth)

        return cat_processing

    def get_uniques_cnt(self, train: SparkDataset, feats: List[str]) -> Series:
        """Get unique values cnt.

        Be aware that this function uses approx_count_distinct and thus cannot return precise results

        Args:
            train: Dataset with train data.
            feats: Features names.

        Returns:
            Series.

        """
        warn_if_not_cached(train.data)

        sdf = train.data.select(feats)

        # TODO SPARK-LAMA: Do we really need this sampling?
        # if self.subsample:
        #     sdf = sdf.sample(withReplacement=False, fraction=self.subsample, seed=self.random_state)

        sdf = sdf.select([F.approx_count_distinct(col).alias(col) for col in feats])
        result = sdf.collect()[0]

        uns = [result[col] for col in feats]
        return Series(uns, index=feats, dtype="int")

    def get_top_categories(self, train: SparkDataset, top_n: int = 5) -> List[str]:
        """Get top categories by importance.

        If feature importance is not defined,
        or feats has same importance - sort it by unique values counts.
        In second case init param ``ascending_by_cardinality``
        defines how - asc or desc.

        Args:
            train: Dataset with train data.
            top_n: Number of top categories.

        Returns:
            List.

        """
        if self.max_intersection_depth <= 1 or self.top_intersections <= 1:
            return []

        cats = get_columns_by_role(train, "Category")
        if len(cats) == 0:
            return []

        df = DataFrame({"importance": 0, "cardinality": 0}, index=cats)
        # importance if defined
        if self.feats_imp is not None:
            feats_imp = Series(self.feats_imp.get_features_score()).sort_values(ascending=False)
            df["importance"] = feats_imp[feats_imp.index.isin(cats)]
            df["importance"].fillna(-np.inf)

        # check for cardinality
        df["cardinality"] = self.get_uniques_cnt(train, cats)
        # sort
        df = df.sort_values(
            by=["importance", "cardinality"],
            ascending=[False, self.ascending_by_cardinality],
        )
        # get top n
        top = list(df.index[:top_n])

        return top


class SparkEmptyFeaturePipeline(SparkFeaturesPipeline):
    """
    This class creates pipeline with ``SparkNoOpTransformer``
    """

    def create_pipeline(self, train: SparkDataset) -> SparkEstOrTrans:
        """
        Returns ``SparkNoOpTransformer`` instance
        """
        return SparkNoOpTransformer()


class SparkNoOpTransformer(SparkBaseTransformer):
    """
    This transformer does nothing, it just returns the input dataframe unchanged.
    """

    def __init__(self):
        super().__init__(input_cols=[], output_cols=[], input_roles=dict(), output_roles=dict())

    def _transform(self, dataset):
        return dataset
