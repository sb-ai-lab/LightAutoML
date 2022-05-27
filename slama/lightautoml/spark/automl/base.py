"""Base AutoML class."""
import logging
from copy import copy
from typing import Any, Callable, Tuple
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

from pyspark.ml import PipelineModel, Transformer
from pyspark.sql import functions as F

from .blend import SparkBlender, SparkBestModelSelector
from ..dataset.base import SparkDataset
from ..pipelines.ml.base import SparkMLPipeline
from ..reader.base import SparkToSparkReader
from ..transformers.base import ColumnsSelectorTransformer
from ..validation.base import SparkBaseTrainValidIterator
from ..validation.iterators import SparkFoldsIterator, SparkHoldoutIterator, SparkDummyIterator
from ...reader.base import RolesDict
from ...utils.logging import set_stdout_level, verbosity_to_loglevel
from ...utils.timer import PipelineTimer

logger = logging.getLogger(__name__)


class SparkAutoML:
    """Class for compile full pipeline of AutoML task.

    AutoML steps:

        - Read, analyze data and get inner
          :class:`~lightautoml.dataset.base.LAMLDataset` from input
          dataset: performed by reader.
        - Create validation scheme.
        - Compute passed ml pipelines from levels.
          Each element of levels is list
          of :class:`~lightautoml.pipelines.ml.base.MLPipelines`
          prediction from current level are passed to next level
          pipelines as features.
        - Time monitoring - check if we have enough time to calc new pipeline.
        - Blend last level models and prune useless pipelines
          to speedup inference: performed by blender.
        - Returns prediction on validation data.
          If crossvalidation scheme is used,
          out-of-fold prediction will returned.
          If validation data is passed
          it will return prediction on validation dataset.
          In case of cv scheme when some point of train data
          never was used as validation (ex. timeout exceeded
          or custom cv iterator like
          :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`
          was used) NaN for this point will be returned.

    Example:
        Common usecase - create custom pipelines or presets.

        >>> reader = SparkToSparkReader()
        >>> pipe = SparkMLPipeline([SparkMLAlgo()])
        >>> levels = [[pipe]]
        >>> automl = SparkAutoML(reader, levels, )
        >>> automl.fit_predict(data, roles={'target': 'TARGET'})

    """

    def __init__(self,
                 reader: SparkToSparkReader,
                 levels: Sequence[Sequence[SparkMLPipeline]],
                 timer: Optional[PipelineTimer] = None,
                 blender: Optional[SparkBlender] = None,
                 skip_conn: bool = False,
                 return_all_predictions: bool = False):
        """

        Args:
            reader: Instance of Reader class object that
              creates :class:`~lightautoml.dataset.base.LAMLDataset`
              from input data.
            levels: List of list
              of :class:`~lightautoml.pipelines.ml..base.MLPipelines`.
            timer: Timer instance of
              :class:`~lightautoml.utils.timer.PipelineTimer`.
              Default - unlimited timer.
            blender: Instance of Blender.
              Default - :class:`~lightautoml.automl.blend.BestModelSelector`.
            skip_conn: True if we should pass first level
              input features to next levels.

        Note:
            There are several verbosity levels:

                - `0`: No messages.
                - `1`: Warnings.
                - `2`: Info.
                - `3`: Debug.

        """
        super().__init__()
        self.levels: Optional[Sequence[Sequence[SparkMLPipeline]]] = None
        self._transformer = None
        self._initialize(reader, levels, timer, blender, skip_conn, return_all_predictions)

    def make_transformer(self, no_reader: bool = False,  return_all_predictions: bool = False) -> Transformer:

        automl_transformer, _ = self._build_transformer(no_reader, return_all_predictions)

        return automl_transformer

    def _initialize(
        self,
        reader: SparkToSparkReader,
        levels: Sequence[Sequence[SparkMLPipeline]],
        timer: Optional[PipelineTimer] = None,
        blender: Optional[SparkBlender] = None,
        skip_conn: bool = False,
        return_all_predictions: bool = False,
    ):
        """Same as __init__. Exists for delayed initialization in presets.

        Args:
            reader: Instance of Reader class object that
              creates :class:`~lightautoml.dataset.base.LAMLDataset`
              from input data.
            levels: List of list
              of :class:`~lightautoml.pipelines.ml..base.MLPipelines`.
            timer: Timer instance of
              :class:`~lightautoml.utils.timer.PipelineTimer`.
              Default - unlimited timer.
            blender: Instance of Blender.
              Default - :class:`~lightautoml.automl.blend.BestModelSelector`.
            skip_conn: True if we should pass first level
              input features to next levels.
            return_all_predictions: True if we should return all predictions from last
              level models.

        """
        assert len(levels) > 0, "At least 1 level should be defined"

        self.timer = timer
        if timer is None:
            self.timer = PipelineTimer()
        self.reader = reader
        self._levels = levels

        # default blender is - select best model and prune other pipes
        self.blender = blender
        if blender is None:
            self.blender = SparkBestModelSelector()

        # update model names
        for i, lvl in enumerate(self._levels):
            for j, pipe in enumerate(lvl):
                pipe.upd_model_names("Lvl_{0}_Pipe_{1}".format(i, j))

        self.skip_conn = skip_conn
        self.return_all_predictions = return_all_predictions

    def fit_predict(
        self,
        train_data: Any,
        roles: dict,
        train_features: Optional[Sequence[str]] = None,
        cv_iter: Optional[Iterable] = None,
        valid_data: Optional[Any] = None,
        valid_features: Optional[Sequence[str]] = None,
        verbose: int = 0,
    ) -> SparkDataset:
        """Fit on input data and make prediction on validation part.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Optional features names,
              if cannot be inferred from train_data.
            cv_iter: Custom cv iterator. For example,
              :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset
              features if can't be inferred from `valid_data`.
            verbose: controls verbosity

        Returns:
            Predicted values.

        """
        set_stdout_level(verbosity_to_loglevel(verbose))
        self.timer.start()

        train_dataset = self.reader.fit_read(train_data, train_features, roles)

        assert (
                len(self._levels) <= 1 or train_dataset.folds is not None
        ), "Not possible to fit more than 1 level without cv folds"

        assert (
                len(self._levels) <= 1 or valid_data is None
        ), "Not possible to fit more than 1 level with holdout validation"

        valid_dataset = self.reader.read(valid_data, valid_features, add_array_attrs=True) if valid_data else None

        train_valid = self._create_validation_iterator(train_dataset, valid_dataset, None, cv_iter=cv_iter)
        current_level_roles = train_valid.train.roles
        pipes: List[SparkMLPipeline] = []
        self.levels = []
        for leven_number, level in enumerate(self._levels, 1):
            pipes = []
            flg_last_level = leven_number == len(self._levels)
            initial_level_roles = current_level_roles

            logger.info(
                f"Layer \x1b[1m{leven_number}\x1b[0m train process start. Time left {self.timer.time_left:.2f} secs"
            )

            level_predictions: Optional[SparkDataset] = None
            for k, ml_pipe in enumerate(level):
                # train_valid = self._create_validation_iterator(level_predictions, None, None, cv_iter=cv_iter)
                train_valid.input_roles = initial_level_roles
                level_predictions = ml_pipe.fit_predict(train_valid)
                level_predictions = self._break_plan(level_predictions)

                train_valid = self._create_validation_iterator(level_predictions, None, None, cv_iter=cv_iter)
                current_level_roles = level_predictions.roles

                pipes.append(ml_pipe)

                logger.info("Time left {:.2f} secs\n".format(self.timer.time_left))

                if self.timer.time_limit_exceeded():
                    logger.info(
                        "Time limit exceeded. Last level models will be blended and unused pipelines will be pruned.\n"
                    )

                    flg_last_level = True
                    break
                else:
                    if self.timer.child_out_of_time:
                        logger.info(
                            "Time limit exceeded in one of the tasks. AutoML will blend level {0} models.\n".format(
                                leven_number
                            )
                        )
                        flg_last_level = True

            logger.info("\x1b[1mLayer {} training completed.\x1b[0m\n".format(leven_number))

            if not flg_last_level:
                self.levels.append(pipes)

            # here is split on exit condition
            if flg_last_level or not self.skip_conn:
                # we leave only prediction columns in the dataframe with this roles
                roles = dict()
                for p in pipes:
                    roles.update(p.output_roles)

                sdf = level_predictions.data.select(*level_predictions.service_columns, *list(roles.keys()))
                level_predictions = level_predictions.empty()
                level_predictions.set_data(sdf, sdf.columns, roles)

        blended_prediction, last_pipes = self.blender.fit_predict(level_predictions, pipes)
        self.levels.append(last_pipes)

        del self._levels

        oof_pred = level_predictions if self.return_all_predictions else blended_prediction
        return oof_pred

    def predict(
        self,
        data: Any,
        features_names: Optional[Sequence[str]] = None,
        return_all_predictions: Optional[bool] = None,
        add_reader_attrs: bool = False
    ) -> SparkDataset:
        """Predict with automl on new dataset.

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
              if cannot be inferred from `train_data`.
            return_all_predictions: if True,
              returns all model predictions from last level
        Returns:
            Dataset with predictions.

        """
        dataset = self.reader.read(data, features_names, add_array_attrs=add_reader_attrs)
        logger.info(f"After Reader in {type(self)}. Columns: {sorted(dataset.data.columns)}")
        automl_transformer, roles = self._build_transformer(no_reader=True, return_all_predictions=return_all_predictions)
        predictions = automl_transformer.transform(dataset.data)

        sds = dataset.empty()
        sds.set_data(predictions, predictions.columns, roles)

        return sds

    def collect_used_feats(self) -> List[str]:
        """Get feats that automl uses on inference.

        Returns:
            Features names list.

        """
        used_feats = set()

        for lvl in self.levels:
            for pipe in lvl:
                used_feats.update(pipe.used_features)

        used_feats = list(used_feats)

        return used_feats

    def collect_model_stats(self) -> Dict[str, int]:
        """Collect info about models in automl.

        Returns:
            Dict with models and its runtime numbers.

        """
        model_stats = {}

        for lvl in self.levels:
            for pipe in lvl:
                for ml_algo in pipe.ml_algos:
                    model_stats[ml_algo.name] = len(ml_algo.models)

        return model_stats

    def _create_validation_iterator(self,
                                    train: SparkDataset,
                                    valid: Optional[SparkDataset],
                                    n_folds: Optional[int],
                                    cv_iter: Optional[Callable]) -> SparkBaseTrainValidIterator:
        if valid:
            dataset = self._merge_train_and_valid_datasets(train, valid)
            iterator = SparkHoldoutIterator(dataset)
        elif cv_iter:
            raise NotImplementedError("Not supported now")
        elif train.folds:
            iterator = SparkFoldsIterator(train, n_folds)
        else:
            iterator = SparkDummyIterator(train)

        logger.info(f"Using train valid iterator of type: {type(iterator)}")

        return iterator

    def _build_transformer(self, no_reader: bool = False,  return_all_predictions: bool = False) \
            -> Tuple[Transformer, RolesDict]:
        stages = []
        if not no_reader:
            stages.append(self.reader.make_transformer(add_array_attrs=True))

        ml_pipes = [ml_pipe.transformer for level in self.levels for ml_pipe in level]
        stages.extend(ml_pipes)

        if not return_all_predictions:
            stages.append(self.blender.transformer)
            output_roles = self.blender.output_roles
        else:
            output_roles = dict()
            for ml_pipe in self.levels[-1]:
                output_roles.update(ml_pipe.output_roles)

        sel_tr = ColumnsSelectorTransformer(
            input_cols=[SparkDataset.ID_COLUMN] + list(output_roles.keys()),
            optional_cols=[self.reader.target_col] if self.reader.target_col else []
        )
        stages.append(sel_tr)

        automl_transformer = PipelineModel(stages=stages)

        return automl_transformer, output_roles

    @staticmethod
    def _merge_train_and_valid_datasets(train: SparkDataset, valid: Optional[SparkDataset]) -> SparkDataset:
        if not valid:
            return train

        assert len(set(train.features).symmetric_difference(set(valid.features))) == 0

        tcols = copy(train.data.columns)
        vcols = copy(valid.data.columns)

        if train.folds_column:
            tcols.remove(train.folds_column)
        if valid.folds_column:
            vcols.remove(valid.folds_column)

        folds_col = train.folds_column if train.folds_column else SparkToSparkReader.DEFAULT_READER_FOLD_COL

        # fold #1
        tdf = train.data.select(*tcols, F.lit(1).alias(folds_col))
        # fold #0
        # In SparkHoldoutIterator we always use fold #0 as validation
        vdf = valid.data.select(*vcols, F.lit(0).alias(folds_col))
        sdf = tdf.unionByName(vdf).coalesce(tdf.rdd.getNumPartitions())

        dataset = train.empty()
        dataset.set_data(sdf, sdf.columns, train.roles)

        return dataset

    @staticmethod
    def _break_plan(train: SparkDataset) -> SparkDataset:
        logger.info("Breaking the plan sequence to reduce wor for optimizer")
        new_df = train.spark_session.createDataFrame(train.data.rdd, schema=train.data.schema, verifySchema=False)

        sds = train.empty()
        sds.set_data(new_df, train.features, train.roles)
        return sds
