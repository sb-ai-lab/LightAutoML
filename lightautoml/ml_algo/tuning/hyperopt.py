"""Classes to implement hyperparameter tuning using Optuna."""

import logging

from copy import copy
from copy import deepcopy
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from yaml import DirectiveToken

from hyperopt import hp, fmin, fmax, tpe
from lightautoml.dataset.base import LAMLDataset
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.ml_algo.tuning.base import Distribution
from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.validation.base import HoldoutIterator
from lightautoml.validation.base import TrainValidIterator


logger = logging.getLogger(__name__)

TunableAlgo = TypeVar("TunableAlgo", bound=MLAlgo)

HYPEROPT_DISTRIBUTIONS_MAP = {
    Distribution.CHOICE: "choice",
    Distribution.UNIFORM: "uniform",
    Distribution.LOGUNIFORM: "loguniform",
    Distribution.INTUNIFORM: "suggest_int",
    Distribution.DISCRETEUNIFORM: "suggest_discrete_uniform",
    Distribution.QUNIFORM: "quniform",
    Distribution.NORMAL: "normal",
    Distribution.QNORMAL: "qnormal",
    Distribution.LOGNORMAL: "lognormal",
}

randint
pchoice
uniformint
qloguniform
qlognormal

class HyperOptTuner(ParamsTuner):
    """Wrapper for HyperOpt tuner.

    Args:
        timeout: Maximum learning time.
        n_trials: Maximum number of trials.
        direction: Direction of optimization.
            Set ``minimize`` for minimization
            and ``maximize`` for maximization.
        fit_on_holdout: Will be used holdout cv-iterator.
        random_state: Seed for optuna sampler.

    """

    _name: str = "HyperOptTuner"

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


        space = self._get_search_space()
        objective = self._get_objective(ml_algo, train_valid_iterator)
        
        if self.direction == 'maximize':
            directed_optimize = fmax
            
        elif self.direction == 'minimize':
            directed_optimize = fmin 

        else:
            raise

        best = directed_optimize(
            objective, 
            space, 
            algo=tpe.suggest, 
            max_evals=self.n_trials
        )
        
        self._best_params = best # hp.space_eval(space, best)
        ml_algo.params = self._best_params
        preds_ds = ml_algo.fit_predict(train_valid_iterator)

        return ml_algo, preds_ds

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
        
        # define an objective function
        def objective(args) -> float:
            _ml_algo = deepcopy(ml_algo)
            _ml_algo.params = args # TODO
            output_dataset = _ml_algo.fit_predict(train_valid_iterator=train_valid_iterator)
            return _ml_algo.score(output_dataset)

        return objective
    
    def _get_search_space(self,):
        hp.choice('a',
            [
                ('case 1', 1 + hp.lognormal('c1', 0, 1)),
                ('case 2', hp.uniform('c2', -10, 10))
            ])

        trial_values = copy(suggested_params)

        for parameter_name, search_space in optimization_search_space.items():
            if search_space.distribution_type in OPTUNA_DISTRIBUTIONS_MAP:
                trial_values[parameter_name] = getattr(
                    trial, 
                    OPTUNA_DISTRIBUTIONS_MAP[search_space.distribution_type]
                )(
                    name=parameter_name, **search_space.params
                )
            else:
                raise ValueError(f"Optuna does not support distribution {search_space.distribution_type}")

        return trial_values
