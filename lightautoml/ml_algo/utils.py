"""Tools for model training."""

import logging

from typing import Optional
from typing import Tuple

from ..dataset.base import LAMLDataset
from ..validation.base import TrainValidIterator
from .base import MLAlgo
from .tuning.base import ParamsTuner


logger = logging.getLogger(__name__)


def tune_and_fit_predict(
    ml_algo: MLAlgo,
    params_tuner: ParamsTuner,
    train_valid: TrainValidIterator,
    force_calc: bool = True,
) -> Tuple[Optional[MLAlgo], Optional[LAMLDataset]]:
    """Tune new algorithm, fit on data and return algo and predictions.

    Args:
        ml_algo: ML algorithm that will be tuned.
        params_tuner: Tuner object.
        train_valid: Classic cv-iterator.
        force_calc: Flag if single fold of ml_algo should be calculated anyway.

    Returns:
        Tuple (BestMlAlgo, predictions).

    """

    timer = ml_algo.timer
    timer.start()
    single_fold_time = timer.estimate_folds_time(1)

    # if force_calc is False we check if it make sense to continue
    if not force_calc and (
        (single_fold_time is not None and single_fold_time > timer.time_left) or timer.time_limit_exceeded()
    ):
        return None, None
    
    if params_tuner.best_params is None:
        # this try/except clause was added because catboost died for some unexpected reason
        try:
            # TODO: Set some conditions to the tuner
            new_algo, preds = params_tuner.fit(ml_algo, train_valid)
        except Exception as e:
            logger.info2("Model {0} failed during params_tuner.fit call.\n\n{1}".format(ml_algo.name, e))
            return None, None

        if preds is not None:
            return new_algo, preds

    if not force_calc and (
        (single_fold_time is not None and single_fold_time > timer.time_left) or timer.time_limit_exceeded()
    ):
        return None, None
    
    ml_algo.params = params_tuner.best_params
    # this try/except clause was added because catboost died for some unexpected reason
    try:
        preds = ml_algo.fit_predict(train_valid)
    except Exception as e:
        logger.info2("Model {0} failed during ml_algo.fit_predict call.\n\n{1}".format(ml_algo.name, e))
        return None, None

    return ml_algo, preds
