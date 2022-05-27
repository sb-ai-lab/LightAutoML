"""Pipeline utils."""

from typing import Any
from typing import List
from typing import Optional
from typing import Sequence

from lightautoml.dataset.base import LAMLDataset


def map_pipeline_names(input_names: Sequence[str], output_names: Sequence[str]) -> List[Optional[str]]:
    """Pipelines create name in the way 'prefix__feature_name'.

    Multiple pipelines will create names
    in the way 'prefix1__prefix2__feature_name'.
    This function maps initial features names to outputs.
    Result may be not exact in some rare cases,
    but it's ok for real pipelines.

    Args:
        input_names: Initial feature names.
        output_names: Output feature names.

    Returns:
        Mapping between feature names.

    """
    # TODO: Add assert here
    mapped: List[Optional[str]] = [None] * len(output_names)
    s_in = set(input_names)

    for n, name in enumerate(output_names):
        splitted = name.split("__")

        for i in range(len(splitted)):
            name = "__".join(splitted[i:])
            if name in s_in:
                mapped[n] = name
                break

    assert None not in mapped, "Can not infer names. For feature selection purposes use simple pipeline (one-to-one)"

    return mapped


def get_columns_by_role(dataset: LAMLDataset, role_name: str, **kwargs: Any) -> List[str]:
    """
    Search for columns with specific role and attributes when building pipeline.

    Args:
        dataset: Dataset to search.
        role_name: Name of features role.
        **kwargs: Specific parameters values to search.
          Example: search for categories with OHE processing only.

    Returns:
        List of str features names.

    """
    features = []
    inv_roles = dataset.inverse_roles
    for role in inv_roles:
        if role.name == role_name:
            flg = True
            # TODO: maybe refactor
            for k in kwargs:
                try:
                    attr = getattr(role, k)
                except AttributeError:
                    flg = False
                    break
                if attr != kwargs[k]:
                    flg = False
                    break
            if flg:
                features.extend(inv_roles[role])

    return sorted(features)
