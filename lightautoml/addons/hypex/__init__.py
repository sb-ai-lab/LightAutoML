"""HypEx Addon for LightAutoML.

This module forwards all imports from the official HypEx package,
maintaining the same API structure as in the original library.

Requirements:
    - Install LightAutoML with HypEx support:
      `pip install lightautoml[hypex]`

Examples:
    Importing models and utilities as in HypEx:

    >>> from lightautoml.addons.hypex import AATest
    >>> from lightautoml.addons.hypex.utils.tutorial_data_creation import create_test_data

    Creating test data:
    >>> some_large_dataframe = create_test_data(
    ...     rs=52, na_step=10, nan_cols=['age', 'gender'], num_users=100_000
    ... )

Raises:
    ImportError: If HypEx is not installed.
"""

import importlib
import sys

MODULE_NAME = "hypex"

try:
    hypex = importlib.import_module(MODULE_NAME)
except ImportError:
    raise ImportError(
        f"{MODULE_NAME} is not installed. Please install it using " f"'pip install lightautoml[{MODULE_NAME}]'."
    )

sys.modules["lightautoml.addons.hypex"] = hypex
