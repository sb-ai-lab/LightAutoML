"""Classes to implement hyperparameter tuning using Optuna."""

import logging

from copy import copy
from copy import deepcopy
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import optuna
from tqdm import tqdm

from ...dataset.base import LAMLDataset
from ..base import MLAlgo
from .base import Choice
from .base import ParamsTuner
from .base import Uniform
from ...validation.base import HoldoutIterator
from ...validation.base import TrainValidIterator
from ...utils.logging import get_stdout_level


logger = logging.getLogger(__name__)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.DEBUG)

TunableAlgo = TypeVar("TunableAlgo", bound=MLAlgo)


class ChoiceWrapOptuna:
    """TODO."""

    def __init__(self, choice) -> None:
        self.choice = choice

    def __call__(self, name, trial):
        """_summary_.

        Args:
            name (_type_): _description_
            trial (_type_): _description_

        Returns:
            _type_: _description_
        """
        return trial.suggest_categorical(name=name, choices=self.choice.options)


class UniformWrapOptuna:
    """TODO."""

    def __init__(self, choice) -> None:
        self.choice = choice

    def __call__(self, name, trial):
        """_summary_.

        Args:
            name (_type_): _description_
            trial (_type_): _description_

        Returns:
            _type_: _description_
        """
        if (self.choice.q is not None) and float(self.choice.q).is_integer() and (self.choice.q == 1):
            result = trial.suggest_int(name=name, low=self.choice.low, high=self.choice.high)
        else:
            result = trial.suggest_float(
                name=name, low=self.choice.low, high=self.choice.high, step=self.choice.q, log=self.choice.log
            )

        return result


OPTUNA_DISTRIBUTIONS_MAP = {Choice: ChoiceWrapOptuna, Uniform: UniformWrapOptuna}


class OptunaTuner(ParamsTuner):
    """Wrapper for optuna tuner.

    Args:
        timeout: Maximum learning time.
        n_trials: Maximum number of trials.
        direction: Direction of optimization.
            Set ``minimize`` for minimization
            and ``maximize`` for maximization.
        fit_on_holdout: Will be used holdout cv-iterator.
        random_state: Seed for optuna sampler.

    """

    _name: str = "OptunaTuner"

    study: optuna.study.Study = None
    estimated_n_trials: int = None
    mean_trial_time: Optional[int] = None

    def __init__(
        # TODO: For now, metric is designed to be greater is better. Change maximize param after metric refactor if needed
        self,
        timeout: Optional[int] = 1000,
        n_trials: Optional[int] = 100,
        direction: Optional[str] = "maximize",
        fit_on_holdout: bool = True,
        random_state: int = 42,
    ):
        self.timeout = timeout
        self.n_trials = n_trials
        self.estimated_n_trials = n_trials
        self.direction = direction
        self._fit_on_holdout = fit_on_holdout
        self.random_state = random_state

    def _upd_timeout(self, timeout):
        self.timeout = min(self.timeout, timeout)

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
            not :class:`~lightautoml.validation.base.HoldoutIterator`.
            Tuple (MlALgo, preds_ds) otherwise.

        """
        assert not ml_algo.is_fitted, "Fitted algo cannot be tuned."
        # optuna.logging.set_verbosity(logger.getEffectiveLevel())
        # upd timeout according to ml_algo timer
        estimated_tuning_time = ml_algo.timer.estimate_tuner_time(len(train_valid_iterator))
        if estimated_tuning_time:
            # TODO: Check for minimal runtime!
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

        # TODO: Check if time estimation will be ok with multiprocessing
        def update_trial_time(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            """Callback for number of iteration with time cut-off.

            Args:
                study: Optuna study object.
                trial: Optuna trial object.

            """
            ml_algo.mean_trial_time = study.trials_dataframe()["duration"].mean().total_seconds()
            self.estimated_n_trials = min(self.n_trials, self.timeout // ml_algo.mean_trial_time)

            logger.info3(
                f"\x1b[1mTrial {len(study.trials)}\x1b[0m with hyperparameters {trial.params} scored {trial.value} in {trial.duration}"
            )

        try:
            # Custom progress bar
            def custom_progress_bar(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
                best_trial = study.best_trial
                progress_bar.set_postfix(best_trial=best_trial.number, best_value=best_trial.value)
                progress_bar.update(1)

            # Initialize progress bar
            if get_stdout_level() in [logging.INFO, logging.INFO2]:
                progress_bar = tqdm(total=self.n_trials, desc="Optimization Progress")

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
                callbacks=(
                    [update_trial_time, custom_progress_bar]
                    if get_stdout_level() in [logging.INFO, logging.INFO2]
                    else [update_trial_time]
                ),
            )

            # Close the progress bar if it was initialized
            if get_stdout_level() in [logging.INFO, logging.INFO2]:
                progress_bar.close()

            # need to update best params here
            self._best_params = self.study.best_params
            ml_algo.params = self._best_params

            logger.info(f"Hyperparameters optimization for \x1b[1m{ml_algo._name}\x1b[0m completed")
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
            ml_algo: Tunable algorithm.
            estimated_n_trials: Maximum number of hyperparameter estimations.
            train_valid_iterator: Used for getting parameters
                depending on dataset.

        Returns:
            Callable objective.

        """
        assert isinstance(ml_algo, MLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            _ml_algo = deepcopy(ml_algo)

            optimization_search_space = _ml_algo.optimization_search_space

            if not optimization_search_space:
                optimization_search_space = _ml_algo._get_default_search_spaces(
                    suggested_params=_ml_algo.init_params_on_input(train_valid_iterator),
                    estimated_n_trials=estimated_n_trials,
                )

            if callable(optimization_search_space):
                _ml_algo.params = optimization_search_space(
                    trial=trial,
                    optimization_search_space=optimization_search_space,
                    suggested_params=_ml_algo.init_params_on_input(train_valid_iterator),
                )
            else:
                _ml_algo.params = self._sample(
                    trial=trial,
                    optimization_search_space=optimization_search_space,
                    suggested_params=_ml_algo.init_params_on_input(train_valid_iterator),
                )

            output_dataset = _ml_algo.fit_predict(train_valid_iterator=train_valid_iterator)

            return _ml_algo.score(output_dataset)

        return objective

    def _sample(
        self,
        optimization_search_space,
        trial: optuna.trial.Trial,
        suggested_params: dict,
    ) -> dict:
        # logger.info3(f'Suggested parameters: {suggested_params}')

        trial_values = copy(suggested_params)

        for parameter_name, search_space in optimization_search_space.items():
            not_supported = True
            for key_class in OPTUNA_DISTRIBUTIONS_MAP:
                if isinstance(search_space, key_class):
                    wrapped_search_space = OPTUNA_DISTRIBUTIONS_MAP[key_class](search_space)
                    trial_values[parameter_name] = wrapped_search_space(
                        name=parameter_name,
                        trial=trial,
                    )
                    not_supported = False
            if not_supported:
                raise ValueError(f"Optuna does not support distribution {search_space}")

        return trial_values

    def plot(self):
        """Plot optimization history of all trials in a study."""
        return optuna.visualization.plot_optimization_history(self.study)


class DLOptunaTuner(ParamsTuner):
    """Wrapper for optuna tuner.

    Args:
        timeout: Maximum learning time.
        n_trials: Maximum number of trials.
        direction: Direction of optimization.
            Set ``minimize`` for minimization
            and ``maximize`` for maximization.
        fit_on_holdout: Will be used holdout cv-iterator.
        random_state: Seed for optuna sampler.

    """

    _name: str = "OptunaTuner"

    study: optuna.study.Study = None
    estimated_n_trials: int = None
    mean_trial_time: Optional[int] = None

    def __init__(
        # TODO: For now, metric is designed to be greater is better. Change maximize param after metric refactor if needed
        self,
        timeout: Optional[int] = 1000,
        n_trials: Optional[int] = 100,
        direction: Optional[str] = "maximize",
        fit_on_holdout: bool = True,
        random_state: int = 42,
    ):
        self.timeout = timeout
        self.n_trials = n_trials
        self.estimated_n_trials = n_trials
        self.direction = direction
        self._fit_on_holdout = fit_on_holdout
        self.random_state = random_state

    def _upd_timeout(self, timeout):
        self.timeout = min(self.timeout, timeout)

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
            not :class:`~lightautoml.validation.base.HoldoutIterator`.
            Tuple (MlALgo, preds_ds) otherwise.

        """
        assert not ml_algo.is_fitted, "Fitted algo cannot be tuned."
        self._params_scores = []

        # optuna.logging.set_verbosity(get_stdout_level())
        # upd timeout according to ml_algo timer
        estimated_tuning_time = ml_algo.timer.estimate_tuner_time(len(train_valid_iterator))
        if estimated_tuning_time:
            # TODO: Check for minimal runtime!
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

        # TODO: Check if time estimation will be ok with multiprocessing
        def update_trial_time(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            """Callback for number of iteration with time cut-off.

            Args:
                study: Optuna study object.
                trial: Optuna trial object.

            """
            ml_algo.mean_trial_time = study.trials_dataframe()["duration"].mean().total_seconds()
            self.estimated_n_trials = min(self.n_trials, self.timeout // ml_algo.mean_trial_time)

            logger.info3(
                f"\x1b[1mTrial {len(study.trials)}\x1b[0m with hyperparameters {trial.params} scored {trial.value} in {trial.duration}"
            )

        try:
            # Custom progress bar
            def custom_progress_bar(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
                best_trial = study.best_trial
                progress_bar.set_postfix(best_trial=best_trial.number, best_value=best_trial.value)
                progress_bar.update(1)

            # Initialize progress bar
            if get_stdout_level() in [logging.INFO, logging.INFO2]:
                progress_bar = tqdm(total=self.n_trials, desc="Optimization Progress")

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
                callbacks=(
                    [update_trial_time, custom_progress_bar]
                    if get_stdout_level() in [logging.INFO, logging.INFO2]
                    else [update_trial_time]
                ),
            )

            # Close the progress bar if it was initialized
            if get_stdout_level() in [logging.INFO, logging.INFO2]:
                progress_bar.close()

            # need to update best params here
            if self.direction == "maximize":
                self._best_params = max(self._params_scores, key=lambda x: x[1])[0]
            else:
                self._best_params = min(self._params_scores, key=lambda x: x[1])[0]

            ml_algo.params = self._best_params
            del self._params_scores

            logger.info(f"Hyperparameters optimization for \x1b[1m{ml_algo._name}\x1b[0m completed")
            logger.info2(
                f"The set of hyperparameters \x1b[1m{self._best_params}\x1b[0m\n achieve {self.study.best_value:.4f} {metric_name}"
            )

            if flg_new_iterator:
                # if tuner was fitted on holdout set we dont need to save train results
                return None, None

            preds_ds = ml_algo.fit_predict(train_valid_iterator)

            return ml_algo, preds_ds
        except optuna.exceptions.OptunaError:
            del self._params_scores
            return None, None

    def _get_objective(
        self,
        ml_algo: TunableAlgo,
        estimated_n_trials: int,
        train_valid_iterator: TrainValidIterator,
    ) -> Callable[[optuna.trial.Trial], Union[float, int]]:
        """Get objective.

        Args:
            ml_algo: Tunable algorithm.
            estimated_n_trials: Maximum number of hyperparameter estimations.
            train_valid_iterator: Used for getting parameters
                depending on dataset.

        Returns:
            Callable objective.

        """
        assert isinstance(ml_algo, MLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            _ml_algo = deepcopy(ml_algo)

            optimization_search_space = _ml_algo.optimization_search_space
            if not optimization_search_space:
                optimization_search_space = _ml_algo._default_sample

            if callable(optimization_search_space):
                sampled_params = optimization_search_space(
                    trial=trial,
                    estimated_n_trials=estimated_n_trials,
                    suggested_params=_ml_algo.init_params_on_input(train_valid_iterator),
                )
            else:
                sampled_params = self._sample(
                    trial=trial,
                    optimization_search_space=optimization_search_space,
                    suggested_params=_ml_algo.init_params_on_input(train_valid_iterator),
                )

            _ml_algo.params = sampled_params
            output_dataset = _ml_algo.fit_predict(train_valid_iterator=train_valid_iterator)
            score = _ml_algo.score(output_dataset)
            self._params_scores.append((sampled_params, score))
            return score

        return objective

    def _sample(
        self,
        optimization_search_space,
        trial: optuna.trial.Trial,
        suggested_params: dict,
    ) -> dict:
        # logger.info3(f'Suggested parameters: {suggested_params}')
        trial_values = copy(suggested_params)
        for parameter_name, search_space in optimization_search_space.items():
            not_supported = True
            for key_class in OPTUNA_DISTRIBUTIONS_MAP:
                if isinstance(search_space, key_class):
                    wrapped_search_space = OPTUNA_DISTRIBUTIONS_MAP[key_class](search_space)
                    trial_values[parameter_name] = wrapped_search_space(
                        name=parameter_name,
                        trial=trial,
                    )
                    not_supported = False
            if not_supported:
                raise ValueError(f"Optuna does not support distribution {search_space}")

    def plot(self):
        """Plot optimization history of all trials in a study."""
        return optuna.visualization.plot_optimization_history(self.study)
