"""Reader utils."""

from typing import Callable
from typing import Optional
from typing import Union

import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from ..tasks import Task


def set_sklearn_folds(
    task: Task,
    target: np.ndarray,
    cv: Union[Callable, int] = 5,
    random_state: int = 42,
    group: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Determines the cross-validation splitting strategy.

    Args:
        task: If `'binary'` or `'multiclass'` used stratified cv.
        target: Target values.
        cv: Specifies number of folds.
        random_state: Determines random number generation.
        group: For group k-folding.

    Returns:
        Array with fold indices.

    """
    if type(cv) is int:
        if group is not None:
            split = GroupKFold(cv).split(group, group, group)
        elif task.name in ["binary", "multiclass"]:

            split = StratifiedKFold(cv, random_state=random_state, shuffle=True).split(target, target)
        else:
            split = KFold(cv, random_state=random_state, shuffle=True).split(target, target)

        folds = np.zeros(target.shape[0], dtype=np.int32)
        for n, (f0, f1) in enumerate(split):
            folds[f1] = n

        return folds

    return
