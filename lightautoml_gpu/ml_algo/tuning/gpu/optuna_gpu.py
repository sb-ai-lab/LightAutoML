""""Classes to implement hyperparameter tuning using Optuna."""

import logging
import multiprocessing
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

import optuna

from lightautoml_gpu.dataset.base import LAMLDataset
from lightautoml_gpu.ml_algo.base import MLAlgo
from lightautoml_gpu.ml_algo.tuning.optuna import (
    OptunaTuner,
    TunableAlgo,
)
from lightautoml_gpu.validation.base import HoldoutIterator, TrainValidIterator

logger = logging.getLogger(__name__)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.DEBUG)


class GpuQueue:
    def __init__(self, ngpus):
        self.ngpus = ngpus
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(self.ngpus)) if self.ngpus > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


class OptunaTunerGPU(OptunaTuner):
    """Wrapper for optuna tuner."""

    _name: str = "OptunaTunerGPU"

    study: optuna.study.Study = None
    estimated_n_trials: int = None
    mean_trial_time: Optional[int] = None

    def __init__(
        self,
        ngpus: int = 1,
        gpu_queue: GpuQueue = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ngpus = ngpus
        self.gpu_queue = gpu_queue

    def fit(
        self,
        ml_algo: TunableAlgo,
        train_valid_iterator: Optional[TrainValidIterator] = None,
    ) -> Tuple[Optional[TunableAlgo], Optional[LAMLDataset]]:
        """Tune model.

        Args:
            ml_algo: Algo that is tuned.
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Tuple (None, None) if an optuna exception raised
            or ``fit_on_holdout=True`` and ``train_valid_iterator`` is
            not :class:`~lightautoml_gpu.validation.base.HoldoutIterator`.
            Tuple (MlALgo, preds_ds) otherwise.

        """
        assert not ml_algo.is_fitted, "Fitted algo cannot be tuned."
        # upd timeout according to ml_algo timer
        estimated_tuning_time = ml_algo.timer.estimate_tuner_time(
            len(train_valid_iterator)
        )
        if estimated_tuning_time:
            estimated_tuning_time = max(estimated_tuning_time, 1)
            self._upd_timeout(estimated_tuning_time)

        logger.info(
            f"Start hyperparameters optimization for \x1b[1m{ml_algo._name}\x1b[0m ... Time budget is {self.timeout:.2f} secs"
        )

        metric_name = train_valid_iterator.train.task.get_dataset_metric().name
        ml_algo = deepcopy(ml_algo)

        flg_new_iterator = False
        if self._fit_on_holdout and type(train_valid_iterator) != HoldoutIterator:
            train_valid_iterator = train_valid_iterator.convert_to_holdout_iterator()
            flg_new_iterator = True

        def update_trial_time(
            study: optuna.study.Study, trial: optuna.trial.FrozenTrial
        ):
            """Callback for number of iteration with time cut-off.

            Args:
                study: Optuna study object.
                trial: Optuna trial object.

            """
            ml_algo.mean_trial_time = (
                study.trials_dataframe()["duration"].mean().total_seconds()
            )
            self.estimated_n_trials = min(
                self.n_trials, self.timeout // ml_algo.mean_trial_time
            )

            logger.info3(
                f"\x1b[1mTrial {len(study.trials)}\x1b[0m with hyperparameters {trial.params} scored {trial.value} in {trial.duration}"
            )

        try:

            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            self.study = optuna.create_study(direction=self.direction, sampler=sampler)

            self.study.optimize(
                func=self._get_objective(
                    ml_algo=ml_algo,
                    estimated_n_trials=self.estimated_n_trials,
                    train_valid_iterator=train_valid_iterator,
                ),
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[update_trial_time],
                n_jobs=self.ngpus,
            )
            # need to update best params here
            self._best_params = self.study.best_params
            ml_algo.params = self._best_params

            logger.info(
                f"Hyperparameters optimization for \x1b[1m{ml_algo._name}\x1b[0m completed"
            )
            logger.info2(
                f"The set of hyperparameters \x1b[1m{self._best_params}\x1b[0m\n achieve {self.study.best_value:.4f} {metric_name}"
            )

            if flg_new_iterator:
                # if tuner was fitted on holdout set we dont need to save train results
                return None, None

            preds_ds = ml_algo.fit_predict(train_valid_iterator)

            return ml_algo, preds_ds
        except optuna.exceptions.OptunaError:
            return None, None

    def _get_objective(
        self,
        ml_algo: TunableAlgo,
        estimated_n_trials: int,
        train_valid_iterator: TrainValidIterator,
    ) -> Callable[[optuna.trial.Trial], Union[float, int]]:
        """Get objective.

        Args:
            estimated_n_trials: Maximum number of hyperparameter estimations.
            train_valid_iterator: Used for getting parameters
              depending on dataset.

        Returns:
            Callable objective.

        """
        assert isinstance(ml_algo, MLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            with self.gpu_queue.one_gpu_per_process() as gpu_id:
                _ml_algo = deepcopy(ml_algo)
                _ml_algo.parallel_folds = False
                _ml_algo.gpu_ids = [gpu_id]

                optimization_search_space = _ml_algo.optimization_search_space

                if not optimization_search_space:
                    optimization_search_space = _ml_algo._get_default_search_spaces(
                        suggested_params=_ml_algo.init_params_on_input(
                            train_valid_iterator
                        ),
                        estimated_n_trials=estimated_n_trials,
                    )

                if callable(optimization_search_space):
                    _ml_algo.params = optimization_search_space(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(
                            train_valid_iterator
                        ),
                    )
                else:
                    _ml_algo.params = self._sample(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(
                            train_valid_iterator
                        ),
                    )
                output_dataset = _ml_algo.fit_predict(
                    train_valid_iterator=train_valid_iterator
                )

                return _ml_algo.score(output_dataset)

        return objective
