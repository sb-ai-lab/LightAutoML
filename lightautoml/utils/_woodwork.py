"""Restore Woodwork accessor caching for AFG on pandas 3.

Only the Woodwork namespaces are changed, but the descriptors belong to pandas
classes and therefore affect every Woodwork user in the process. Importing this
module has no side effects; AFG constructors and Dask worker preloads opt in.
"""

from importlib.metadata import version
from threading import Lock

import pandas as pd

from packaging.version import Version


_lock = Lock()


class _CachedWoodworkAccessor:
    """Cache an accessor on its owner, as pandas did before version 3."""

    def __init__(self, accessor):
        self._accessor = accessor

    def __get__(self, obj, cls=None):
        if obj is None:
            return self._accessor
        accessor = self._accessor(obj)
        object.__setattr__(obj, "ww", accessor)
        return accessor


def initialize_woodwork():
    """Enable the workaround for pandas 3 and Woodwork releases through 0.31.

    Return whether this process needs the workaround. Pandas 2 and future
    Woodwork releases keep their original descriptors. Repeated calls are safe.
    """
    if Version(pd.__version__).major != 3 or Version(version("woodwork")) >= Version("0.32"):
        return False

    # Importing Woodwork registers both accessors before we replace them.
    import woodwork  # noqa: F401

    with _lock:
        for owner in (pd.DataFrame, pd.Series):
            descriptor = vars(owner)["ww"]
            if not isinstance(descriptor, _CachedWoodworkAccessor):
                setattr(owner, "ww", _CachedWoodworkAccessor(descriptor._accessor))
    return True


def dask_setup(worker):
    """Initialize each Dask worker before it deserializes an EntitySet."""
    initialize_woodwork()
