"""Reader utils."""

from typing import Callable
from typing import Optional
from typing import Union

import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import torch
if torch.cuda.is_available():
    import cudf
    import cupy as cp
    import dask_cudf
else:
    print("could not load gpu related libs (reader/utils.py)")

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
    # GPU PART
    if (torch.cuda.is_available() and isinstance(target, (cp.ndarray, cudf.Series,
                            cudf.DataFrame, dask_cudf.Series, dask_cudf.DataFrame)
                   )):
        def KFolds_gpu(
            target: Union[cudf.Series, cudf.DataFrame],
            n_splits: int = 5,
            shuffle: bool = True,
            random_state: int = 42,
        ) -> cudf.Series:
            """Performs regular KFolds on GPU

            Args:
                target: Target values.
                shuffle: If data needs to shuffled.
                random_state: Determines random number generation.
                n_splits: Number of splits

            Returns:
                Array with fold indices.

            """
            cp.random.seed(seed=random_state)
            n_samples = len(target)
            indices = cp.arange(n_samples)
            if shuffle:
                cp.random.shuffle(indices)
            fold_sizes = cp.full(n_splits, n_samples // n_splits, dtype=int)
            fold_sizes[: n_samples % n_splits] += 1
            current = 0
            output = cp.zeros(n_samples, dtype="i")
            for i, fold_size in enumerate(fold_sizes):
                start, stop = current, current + fold_size
                output[indices[start:stop]] = i
                current = stop
            output = cudf.Series(output, index=target.index, name="folds")
            return output

        if type(cv) is int:
            output = None
            if isinstance(target, (dask_cudf.Series, dask_cudf.DataFrame)):
                shuffle = True
                output = target.map_partitions(
                    KFolds_gpu, cv, shuffle, random_state, meta=("folds", np.int32)
                ).persist()

            elif group is not None:
                n_samples = len(target)
                n_splits = cv
                unique_groups, groups = cp.unique(group, return_inverse=True)
                n_samples_per_group = cp.bincount(groups)
                indices = cp.argsort(n_samples_per_group)[::-1]
                n_samples_per_group = n_samples_per_group[indices]
                n_samples_per_fold = cp.zeros(n_splits)
                group_to_fold = cp.zeros(len(unique_groups))
                for group_index, weight in enumerate(n_samples_per_group):
                    lightest_fold = cp.argmin(n_samples_per_fold)
                    n_samples_per_fold[lightest_fold] += weight
                    group_to_fold[indices[group_index]] = lightest_fold

                indices = group_to_fold[groups]
                output = cp.zeros(n_samples, dtype="i")
                for i in range(n_splits):
                    output[cp.where(indices == i)] = i
                output = cudf.Series(output, index=target.index)

            elif task.name in ["binary", "multiclass"]:
                cp.random.seed(seed=42)
                shuffle = True
                n_splits = cv
                _, y_idx, y_inv = cp.unique(
                    target, return_index=True, return_inverse=True
                )
                _, class_perm = cp.unique(y_idx, return_inverse=True)
                y_encoded = class_perm[y_inv]
                n_classes = len(y_idx)
                y_order = cp.sort(y_encoded)

                allocation = cp.asarray(
                    [
                        cp.bincount(y_order[i::n_splits], minlength=n_classes)
                        for i in range(n_splits)
                    ]
                ).get()

                output = cp.empty(len(target), dtype="i")
                for k in range(n_classes):
                    folds_for_class = cp.arange(n_splits).repeat(
                        allocation[:, k].tolist()
                    )

                    if shuffle:
                        cp.random.shuffle(folds_for_class)
                    output[y_encoded == k] = folds_for_class
                output = cudf.Series(output, index=target.index, name="folds")

            else:
                shuffle = True
                output = KFolds_gpu(target, cv, shuffle, random_state)

            return output
    # CPU PART
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
