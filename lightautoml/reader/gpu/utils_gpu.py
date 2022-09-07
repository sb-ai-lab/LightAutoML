"""Reader utils on gpu."""

from typing import Optional, Union, Callable

import numpy as np
import cupy as cp
import cudf
import dask_cudf

from ...tasks import Task

GpuSeries = Union[cp.ndarray, cudf.Series, dask_cudf.Series]

def set_sklearn_folds_gpu(task: Task, target: GpuSeries, cv: Union[Callable, int] = 5,
                      random_state: int = 42, group: Optional[np.ndarray] = None)-> GpuSeries:
    """Determines the cross-validation splitting strategy. If target is dask_cudf.Series
       then regular KFolds is used

    Args:
        task: If `'binary'` or `'multiclass'` use stratified cv.
        target: Target values.
        cv: Specifies number of folds.
        random_state: Determines random number generation.
        group: For group k-folding.

    Returns:
        Array with fold indices.

    """
    def KFolds_gpu(target: cudf.Series, n_splits: int = 5, shuffle: bool = True,
                   random_state: int = 42) -> cudf.Series:
        """Performs regular KFolds

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
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        output = cp.zeros(n_samples, dtype='i')
        for i, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            output[indices[start:stop]] = i
            current = stop
        output = cudf.Series(output, index=target.index, name='folds')
        return output

    if type(cv) is int:
        output = None
        if isinstance(target, (dask_cudf.Series, dask_cudf.DataFrame)):
            shuffle = True
            output = target.map_partitions(KFolds_gpu, cv, shuffle,
                                           random_state, meta=('folds', np.int32))

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
            output = cp.zeros(n_samples, dtype='i')
            for i in range(n_splits):
                output[cp.where(indices==i)] = i
            output = cudf.Series(output, index=target.index)

        elif task.name in ['binary', 'multiclass']:
            cp.random.seed(seed=42)
            shuffle = True
            n_splits = cv
            _, y_idx, y_inv = cp.unique(target, return_index=True, return_inverse=True)
            _, class_perm = cp.unique(y_idx, return_inverse=True)
            y_encoded = class_perm[y_inv]
            n_classes = len(y_idx)
            y_order = cp.sort(y_encoded)

            allocation = cp.asarray(
                    [cp.bincount(y_order[i::n_splits], minlength=n_classes)
                    for i in range(n_splits)]).get()

            output = cp.empty(len(target), dtype='i')
            for k in range(n_classes):
                folds_for_class = cp.arange(n_splits).repeat(allocation[:, k].tolist())

                if shuffle:
                    cp.random.shuffle(folds_for_class)
                output[y_encoded == k] = folds_for_class
            output = cudf.Series(output, index=target.index)

        else:
            shuffle=True
            output = KFolds_gpu(target, cv, shuffle, random_state)

        return output

    return
