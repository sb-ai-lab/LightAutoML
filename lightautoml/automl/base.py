"""Base AutoML class."""

import logging

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

from ..dataset.base import LAMLDataset
from ..dataset.utils import concatenate
from ..pipelines.ml.base import MLPipeline
from ..reader.base import Reader
from ..utils.logging import set_stdout_level
from ..utils.logging import verbosity_to_loglevel
from ..utils.timer import PipelineTimer
from ..validation.utils import create_validation_iterator
from .blend import BestModelSelector
from .blend import Blender


logger = logging.getLogger(__name__)


class AutoML:
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

        >>> reader = SomeReader()
        >>> pipe = MLPipeline([SomeAlgo()])
        >>> levels = [[pipe]]
        >>> automl = AutoML(reader, levels, )
        >>> automl.fit_predict(data, roles={'target': 'TARGET'})

    """

    def __init__(
        self,
        reader: Reader,
        levels: Sequence[Sequence[MLPipeline]],
        timer: Optional[PipelineTimer] = None,
        blender: Optional[Blender] = None,
        skip_conn: bool = False,
        return_all_predictions: bool = False,
    ):
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
        self._initialize(reader, levels, timer, blender, skip_conn, return_all_predictions)

    def _initialize(
        self,
        reader: Reader,
        levels: Sequence[Sequence[MLPipeline]],
        timer: Optional[PipelineTimer] = None,
        blender: Optional[Blender] = None,
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
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;

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
            self.blender = BestModelSelector()

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
    ) -> LAMLDataset:
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

        valid_dataset = None
        if valid_data is not None:
            valid_dataset = self.reader.read(valid_data, valid_features, add_array_attrs=True)

        train_valid = create_validation_iterator(train_dataset, valid_dataset, n_folds=None, cv_iter=cv_iter)
        # for pycharm)
        level_predictions = None
        pipes = None

        self.levels = []

        for leven_number, level in enumerate(self._levels, 1):
            pipes = []
            level_predictions = []
            flg_last_level = leven_number == len(self._levels)

            logger.info(
                f"Layer \x1b[1m{leven_number}\x1b[0m train process start. Time left {self.timer.time_left:.2f} secs"
            )

            for k, ml_pipe in enumerate(level):

                pipe_pred = ml_pipe.fit_predict(train_valid)
                level_predictions.append(pipe_pred)
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

            # here is split on exit condition
            if not flg_last_level:

                self.levels.append(pipes)
                level_predictions = concatenate(level_predictions)

                if self.skip_conn:
                    valid_part = train_valid.get_validation_data()
                    try:
                        # convert to initital dataset type
                        level_predictions = valid_part.from_dataset(level_predictions)
                    except TypeError:
                        raise TypeError(
                            "Can not convert prediction dataset type to input features. Set skip_conn=False"
                        )
                    level_predictions = concatenate([level_predictions, valid_part])
                train_valid = create_validation_iterator(level_predictions, None, n_folds=None, cv_iter=None)
            else:
                break

        blended_prediction, last_pipes = self.blender.fit_predict(level_predictions, pipes)
        self.levels.append(last_pipes)

        self.reader.upd_used_features(remove=list(set(self.reader.used_features) - set(self.collect_used_feats())))

        del self._levels

        if self.return_all_predictions:
            return concatenate(level_predictions)
        return blended_prediction

    def predict(
        self,
        data: Any,
        features_names: Optional[Sequence[str]] = None,
        return_all_predictions: Optional[bool] = None,
    ) -> LAMLDataset:
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
        dataset = self.reader.read(data, features_names=features_names, add_array_attrs=False)

        for n, level in enumerate(self.levels, 1):
            # check if last level

            level_predictions = []
            for _n, ml_pipe in enumerate(level):
                level_predictions.append(ml_pipe.predict(dataset))

            if n != len(self.levels):

                level_predictions = concatenate(level_predictions)

                if self.skip_conn:

                    try:
                        # convert to initital dataset type
                        level_predictions = dataset.from_dataset(level_predictions)
                    except TypeError:
                        raise TypeError(
                            "Can not convert prediction dataset type to input features. Set skip_conn=False"
                        )
                    dataset = concatenate([level_predictions, dataset])
                else:
                    dataset = level_predictions
            else:
                if (return_all_predictions is None and self.return_all_predictions) or return_all_predictions:
                    return concatenate(level_predictions)
                return self.blender.predict(level_predictions)

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
