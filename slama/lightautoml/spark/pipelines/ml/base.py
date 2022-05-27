"""Base classes for MLPipeline."""
import uuid
import warnings
from copy import copy
from typing import List, cast, Sequence, Union, Tuple, Optional

from pyspark.ml import Transformer, PipelineModel

from ..base import OutputFeaturesAndRoles
from ..features.base import SparkFeaturesPipeline, SparkEmptyFeaturePipeline
from ...dataset.base import LAMLDataset, SparkDataset
from ...ml_algo.base import SparkTabularMLAlgo
from ...transformers.base import ColumnsSelectorTransformer
from ...utils import Cacher
from ...validation.base import SparkBaseTrainValidIterator
from ....ml_algo.tuning.base import ParamsTuner
from ....ml_algo.utils import tune_and_fit_predict
from ....pipelines.ml.base import MLPipeline as LAMAMLPipeline
from ....pipelines.selection.base import SelectionPipeline


class SparkMLPipeline(LAMAMLPipeline, OutputFeaturesAndRoles):
    """Spark version of :class:`~lightautoml.pipelines.ml.base.MLPipeline`. Single ML pipeline.

    Merge together stage of building ML model
    (every step, excluding model training, is optional):

        - Pre selection: select features from input data.
          Performed by
          :class:`~lightautoml.pipelines.selection.base.SelectionPipeline`.
        - Features generation: build new features from selected.
          Performed by
          :class:`~lightautoml.spark.pipelines.features.base.SparkFeaturesPipeline`.
        - Post selection: One more selection step - from created features.
          Performed by
          :class:`~lightautoml.pipelines.selection.base.SelectionPipeline`.
        - Hyperparams optimization for one or multiple ML models.
          Performed by
          :class:`~lightautoml.ml_algo.tuning.base.ParamsTuner`.
        - Train one or multiple ML models:
          Performed by :class:`~lightautoml.spark.ml_algo.base.SparkTabularMLAlgo`.
          This step is the only required for at least 1 model.

    """

    def __init__(
        self,
        cacher_key: str,
        ml_algos: Sequence[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]],
        force_calc: Union[bool, Sequence[bool]] = True,
        pre_selection: Optional[SelectionPipeline] = None,
        features_pipeline: Optional[SparkFeaturesPipeline] = None,
        post_selection: Optional[SelectionPipeline] = None,
        name: Optional[str] = None
    ):
        if features_pipeline is None:
            features_pipeline = SparkEmptyFeaturePipeline()

        super().__init__(ml_algos, force_calc, pre_selection, features_pipeline, post_selection)

        self._cacher_key = cacher_key
        self._output_features = None
        self._output_roles = None
        self._transformer: Optional[Transformer] = None
        self._name = name if name else str(uuid.uuid4())[:5]
        self.ml_algos: List[SparkTabularMLAlgo] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def transformer(self):
        assert self._transformer is not None, f"{type(self)} seems to be not fitted"

        return self._transformer

    def fit_predict(self, train_valid: SparkBaseTrainValidIterator) -> LAMLDataset:
        """Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: Dataset iterator.

        Returns:
            Dataset with predictions of all models.

        """

        # train and apply pre selection
        input_roles = copy(train_valid.input_roles)
        full_train_roles = copy(train_valid.train.roles)

        train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)
        fp = cast(SparkFeaturesPipeline, self.features_pipeline)

        # train and apply post selection
        train_valid = train_valid.apply_selector(self.post_selection)

        preds: Optional[SparkDataset] = None
        for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc):
            ml_algo = cast(SparkTabularMLAlgo, ml_algo)
            ml_algo, curr_preds = tune_and_fit_predict(ml_algo, param_tuner, train_valid, force_calc)
            if ml_algo is not None:
                self.ml_algos.append(ml_algo)
                preds = curr_preds
                train_valid.train = preds
            else:
                warnings.warn("Current ml_algo has not been trained by some reason. "
                              "Check logs for more details.", RuntimeWarning)

        assert (
            len(self.ml_algos) > 0
        ), "Pipeline finished with 0 models for some reason.\nProbably one or more models failed"

        del self._ml_algos

        self._output_roles = {ml_algo.prediction_feature: ml_algo.prediction_role
                              for ml_algo in self.ml_algos}

        # all out roles for the output dataset
        out_roles = copy(self._output_roles)
        # we need also add roles for predictions of previous pipe in this layer
        # because they are not part of either output roles of this pipe
        # (pipes work only with input data to the layer itself, e.g. independently)
        # nor input roles of train_valid iterator
        # (for each pipe iterator represent only input columns to the layer,
        # not outputs of other ml pipes in the layer)
        out_roles.update(full_train_roles)
        # we also need update our out_roles with input_roles to replace roles of input of the layer
        # in case they were changed by SparkChangeRolesTransformer
        out_roles.update(input_roles)

        select_transformer = ColumnsSelectorTransformer(
            input_cols=[SparkDataset.ID_COLUMN, *list(out_roles.keys())],
            optional_cols=[train_valid.train.target_column] if train_valid.train.target_column else []
        )
        ml_algo_transformers = PipelineModel(stages=[ml_algo.transformer for ml_algo in self.ml_algos])
        self._transformer = PipelineModel(stages=[fp.transformer, ml_algo_transformers, select_transformer])

        val_preds = [preds.data]
        val_preds_df = train_valid.combine_val_preds(val_preds, include_train=False)
        val_preds_df = val_preds_df.select(
            SparkDataset.ID_COLUMN,
            train_valid.train.target_column,
            train_valid.train.folds_column,
            *list(out_roles.keys())
        )

        cacher = Cacher(key=self._cacher_key)
        cacher.fit(val_preds_df)
        val_preds_df = cacher.dataset
        val_preds_ds = train_valid.train.empty()
        val_preds_ds.set_data(val_preds_df, list(out_roles.keys()), out_roles)

        return val_preds_ds

    def predict(self, dataset: SparkDataset) -> SparkDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predictions of all trained models.

        """
        out_sdf = self.transformer.transform(dataset.data)

        out_roles = copy(dataset.roles)
        out_roles.update(self.output_roles)

        out_ds = dataset.empty()
        out_ds.set_data(out_sdf, list(out_roles.keys()), out_roles)

        return out_ds
