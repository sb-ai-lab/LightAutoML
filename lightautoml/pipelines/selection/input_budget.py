"""Select a shared set of input columns before individual pipelines are fitted."""

from numbers import Integral

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor


def validate_feature_count(name, value):
    """Require a positive integral feature count."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def select_input_features(dataset, limit, treatment=None):
    """Rank training inputs by tree importance, averaging the two arm rankings for uplift.

    Forced inputs consume the budget first. Each arm contributes equally regardless
    of its size. Ties are resolved in input column order.
    """
    validate_feature_count("max_input_features", limit)
    names = dataset.features
    roles = dataset.roles
    forced = [name for name in names if roles[name].force_input]
    if len(forced) > limit:
        raise ValueError("Forced features exceed max_input_features")
    if len(names) <= limit:
        return list(names)
    if len(forced) == limit:
        return forced

    frame = dataset.to_pandas().data
    values = np.empty(frame.shape, dtype=np.float32)
    for i, name in enumerate(names):
        col = frame[name]
        if roles[name].name == "Numeric":
            values[:, i] = col.to_numpy(dtype=np.float32, na_value=np.nan)
        elif pd.api.types.is_datetime64_any_dtype(col):
            values[:, i] = (col - col.min()).dt.total_seconds()
        else:
            values[:, i] = pd.factorize(col)[0]
    # Older sklearn versions require finite values.
    values[~np.isfinite(values)] = 0

    estimator = ExtraTreesRegressor if dataset.task.name in ("reg", "multi:reg") else ExtraTreesClassifier
    y = np.asarray(dataset.target)
    weights = None if dataset.weights is None else np.asarray(dataset.weights)
    groups = [slice(None)] if treatment is None else [np.asarray(treatment) == arm for arm in (0, 1)]
    importance = np.zeros(len(names))
    for rows in groups:
        model = estimator(n_estimators=32, max_depth=6, random_state=42, n_jobs=1)
        model.fit(values[rows], y[rows], sample_weight=None if weights is None else weights[rows])
        importance += model.feature_importances_
    mandatory = set(forced)
    ranked = [names[i] for i in np.argsort(-importance, kind="stable") if names[i] not in mandatory]
    return forced + ranked[: limit - len(forced)]


def restrict_training_inputs(data, roles, reader, limit):
    """Select a common training frame for multiple learners, keeping auxiliary columns."""
    dataset = reader.fit_read(data, roles=roles)
    selected = set(select_input_features(dataset, limit, treatment=dataset.treatment))
    for columns in reader.used_array_attrs.values():
        selected.update([columns] if isinstance(columns, str) else columns)
    kept_roles = {}
    for role, columns in roles.items():
        if isinstance(columns, str):
            if columns in selected:
                kept_roles[role] = columns
        else:
            kept = [col for col in columns if col in selected]
            if kept:
                kept_roles[role] = kept
    return data.loc[:, [col for col in data if col in selected]], kept_roles
