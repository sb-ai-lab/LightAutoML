"""Utilities for working with the structure of a dataset."""

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.np_pd_dataset import CSRSparseDataset
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import ColumnRole


# RoleType = TypeVar("RoleType", bound=ColumnRole)


def roles_parser(init_roles: Dict[Union[ColumnRole, str], Union[str, Sequence[str]]]) -> Dict[str, ColumnRole]:
    """Parser of roles.

    Parse roles from old format numeric:
    ``[var1, var2, ...]`` to ``{var1:numeric, var2:numeric, ...}``.

    Args:
        init_roles: Mapping between roles and feature names.

    Returns:
        Roles dict in format key - feature names, value - roles.

    """
    roles = {}
    for r in init_roles:

        feat = init_roles[r]

        if isinstance(feat, str):
            roles[feat] = r

        else:
            for f in init_roles[r]:
                roles[f] = r

    return roles


def get_common_concat(
    datasets: Sequence[LAMLDataset],
) -> Tuple[Callable, Optional[type]]:
    """Get concatenation function for datasets of different types.

    Takes multiple datasets as input and check,
    if is's ok to concatenate it and return function.

    Args:
        datasets: Sequence of datasets.

    Returns:
        Function, that is able to concatenate datasets.

    """
    # TODO: Add pandas + numpy via transforming to numpy?
    dataset_types = set([type(x) for x in datasets])

    # general - if single type, concatenation for that type
    if len(dataset_types) == 1:
        klass = list(dataset_types)[0]
        return klass.concat, None

    # np and sparse goes to sparse
    elif dataset_types == {NumpyDataset, CSRSparseDataset}:
        return CSRSparseDataset.concat, CSRSparseDataset

    elif dataset_types == {NumpyDataset, PandasDataset}:
        return numpy_and_pandas_concat, None

    raise TypeError("Unable to concatenate dataset types {0}".format(list(dataset_types)))


def numpy_and_pandas_concat(datasets: Sequence[Union[NumpyDataset, PandasDataset]]) -> PandasDataset:
    """Concat of numpy and pandas dataset.

    Args:
        datasets: Sequence of datasets to concatenate.

    Returns:
        Concatenated dataset.

    """
    datasets = [x.to_pandas() for x in datasets]

    return PandasDataset.concat(datasets)


def concatenate(datasets: Sequence[LAMLDataset]) -> LAMLDataset:
    """Dataset concatenation function.

    Check if datasets have common concat function and then apply.
    Assume to take target/folds/weights etc from first one.

    Args:
        datasets: Sequence of datasets.

    Returns:
        Dataset with concatenated features.

    """
    conc, klass = get_common_concat([ds for ds in datasets if ds is not None])

    # this part is made to avoid setting first dataset of required type
    if klass is not None:

        n = 0
        for n, ds in enumerate(datasets):
            if type(ds) is klass:
                break

        datasets = [datasets[n]] + [x for (y, x) in enumerate(datasets) if n != y]

    return conc(datasets)
