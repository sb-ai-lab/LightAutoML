"""Base classes to implement hyperparameter tuning."""

from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import overload

from lightautoml.dataset.base import LAMLDataset

# if TYPE_CHECKING:
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.validation.base import TrainValidIterator


class Distribution(Enum):
    CHOICE = 0

    UNIFORM = 1
    INTUNIFORM = 2
    QUNIFORM = 3
    LOGUNIFORM = 4
    DISCRETEUNIFORM = 5

    NORMAL = 6
    QNORMAL = 7
    LOGNORMAL = 8


class SearchSpace:
    distribution_type: Distribution = None
    params: Dict = {}

    def __init__(self, distribution_type: Distribution, *args, **kwargs):
        self.distribution_type = distribution_type
        self.params = kwargs


class ParamsTuner(ABC):
    """Base abstract class for hyperparameters tuners."""

    _name: str = "AbstractTuner"
    _best_params: Dict = None
    _fit_on_holdout: bool = False  # if tuner should be fitted on holdout set

    @property
    def best_params(self) -> dict:
        """Get best params.

        Returns:
            Dict with best fitted params.

        """
        assert hasattr(self, "_best_params"), "ParamsTuner should be fitted first"
        return self._best_params

    @overload
    def fit(
        self,
        ml_algo: "MLAlgo",
        train_valid_iterator: Optional[TrainValidIterator] = None,
    ) -> Tuple["MLAlgo", LAMLDataset]:
        ...

    @abstractmethod
    def fit(
        self,
        ml_algo: "MLAlgo",
        train_valid_iterator: Optional[TrainValidIterator] = None,
    ) -> Tuple[None, None]:
        """Tune model hyperparameters.

        Args:
            ml_algo: ML algorithm.
            train_valid_iterator: Classic cv-iterator.

        Returns:
            (None, None) if ml_algo is fitted or models are not fitted during training,
            (BestMLAlgo, BestPredictionsLAMLDataset) otherwise.

        """


class DefaultTuner(ParamsTuner):
    """Default realization of ParamsTuner - just take algo's defaults."""

    _name: str = "DefaultTuner"

    def fit(
        self,
        ml_algo: "MLAlgo",
        train_valid_iterator: Optional[TrainValidIterator] = None,
    ) -> Tuple[None, None]:
        """
        Default fit method - just save defaults.

        Args:
            ml_algo: Algorithm that is tuned.
            train_valid_iterator: Empty.

        Returns:s
            Tuple (None, None).
        """
        self._best_params = ml_algo.init_params_on_input(train_valid_iterator=train_valid_iterator)
        return None, None
