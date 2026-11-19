import calendar
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)


def calc_one_feat_imp(
    iters, level, feat, model, data, norm_score, target, metric, silent, *, permutation=None, weights=None
):
    """Measure one permutation on the inference pipeline starting at the given level."""
    if permutation is None:
        permutation = np.random.RandomState(42).permutation(data.shape[0])
    original = data[:, feat].data.copy()
    try:
        data[feat] = np.asarray(original)[permutation]
        preds = model._predict_from_level(data, level)
    finally:
        data[feat] = original
    preds.target = np.asarray(target)
    preds.weights = data.weights if weights is None else weights
    importance = norm_score - metric(preds)
    if not silent:
        logger.info3(f"{iters[0]}/{iters[1]} permutation importance for {feat}: {importance:.7f}")
    return feat, importance


def calc_feats_permutation_imps(model, used_feats, data, target, metric, silent=False, random_state=42):
    """Score each input at the first level where it is available, including stacking inputs."""
    frame = data.reset_index(drop=True)
    labeled = model.reader.read(frame, add_array_attrs=True)
    if labeled.target is None:
        raise ValueError(f"Permutation importance requires labeled data with target {target}")
    labels = np.asarray(labeled.target)
    weights = None if labeled.weights is None else np.asarray(labeled.weights)
    del labeled
    # Run exactly the serving path: scoring labels and weights stay outside the predictors.
    data = model.reader.read(frame)
    preds = model._predict_from_level(data, 0)
    preds.target = labels
    preds.weights = weights
    normal_score = metric(preds)
    permutation = np.random.RandomState(random_state).permutation(data.shape[0])
    remaining = set(used_feats)
    importances = []
    for level, pipes in enumerate(model.levels):
        for feature in data.features:
            if feature not in remaining:
                continue
            result = calc_one_feat_imp(
                (len(importances) + 1, len(used_feats)),
                level,
                feature,
                model,
                data,
                normal_score,
                labels,
                metric,
                silent,
                permutation=permutation,
                weights=weights,
            )
            remaining.remove(feature)
            importances.append(result)
        if level + 1 < len(model.levels):
            data = model._next_level_input(data, [pipe.predict(data) for pipe in pipes])

    return (
        pd.DataFrame(importances, columns=["Feature", "Importance"])
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def change_datetime(feature_datetime, key, value):
    assert key in ["year", "month", "dayofweek"]
    MAX_DAY = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    changed = []
    if key == "year":
        year = value
        for i in feature_datetime:
            if i.month == 2 and i.day == 29 and not calendar.isleap(year):
                i -= pd.Timedelta(1, "d")
            changed.append(i.replace(year=year))
    if key == "month":
        month = value
        assert month in np.arange(1, 13)
        for i in feature_datetime:
            if i.day > MAX_DAY[month]:
                i -= pd.Timedelta(i.day - MAX_DAY[month], "d")
                if month == 2 and i.day == 28 and calendar.isleap(i.year):
                    i += pd.Timedelta(1, "d")
            changed.append(i.replace(month=month))
    if key == "dayofweek":
        dayofweek = value
        assert value in np.arange(7)
        for i in feature_datetime:
            i += pd.Timedelta(dayofweek - i.dayofweek, "d")
            changed.append(i)
    return np.array(changed)


def plot_pdp_with_distribution(
    test_data,
    grid,
    ys,
    counts,
    reader,
    feature_name,
    individual,
    top_n_classes,
    datetime_level,
):
    feature_role = reader._roles[feature_name].name
    # I. Plot pdp
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})
    axs[0].set_title("PDP: " + feature_name)
    n_classes = ys[0].shape[1]
    if n_classes == 1:
        data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "x": grid[i],
                        "y": ys[i].to_numpy(),
                        "unit": np.arange(ys[i].shape[0]),
                    }
                )
                for i, _ in enumerate(grid)
            ]
        ).reset_index(drop=True)
        if feature_role in ["Numeric", "Datetime"]:
            if individual:
                g0 = sns.lineplot(
                    data=data[data["unit"] > 0],
                    x="x",
                    y="y",
                    ax=axs[0],
                    units="unit",
                    estimator=None,
                    alpha=0.05,
                    color="b",
                )
                g0 = sns.lineplot(
                    data=data[data["unit"] == 0],
                    x="x",
                    y="y",
                    ax=axs[0],
                    alpha=0.05,
                    color="b",
                    label="individual PDP",
                )
                g0 = sns.lineplot(
                    x=grid,
                    y=data.groupby("x").mean()["y"].values,
                    ax=axs[0],
                    linewidth=2,
                    color="r",
                    label="mean PDP",
                )
            else:
                g0 = sns.lineplot(data=data, x="x", y="y", ax=axs[0], color="b")
        else:
            g0 = sns.boxplot(data=data, x="x", y="y", ax=axs[0], showfliers=False, color="b")
    else:
        if reader.class_mapping:
            classes = sorted(reader.class_mapping, key=reader.class_mapping.get)[:top_n_classes]
        else:
            classes = np.arange(min(n_classes, top_n_classes))
        data = pd.concat(
            [
                pd.DataFrame({"x": grid[i], "y": ys[i][:, k], "class": name})
                for i, _ in enumerate(grid)
                for k, name in enumerate(classes)
            ]
        ).reset_index(drop=True)
        if reader._roles[feature_name].name in ["Numeric", "Datetime"]:
            g0 = sns.lineplot(data=data, x="x", y="y", hue="class", ax=axs[0])
        else:
            g0 = sns.boxplot(data=data, x="x", y="y", hue="class", ax=axs[0], showfliers=False)
    g0.set(ylabel="y_pred")
    # II. Plot distribution
    counts = np.array(counts) / sum(counts)
    if feature_role == "Numeric":
        g0.set(xlabel="feature value")
        g1 = sns.histplot(test_data[feature_name], kde=True, color="gray", ax=axs[1])
    elif feature_role == "Category":
        g0.set(xlabel=None)
        axs[0].set_xticklabels(grid, rotation=90)
        g1 = sns.barplot(x=grid, y=counts, ax=axs[1], color="gray")
    else:
        g0.set(xlabel=datetime_level)
        g1 = sns.barplot(x=grid, y=counts, ax=axs[1], color="gray")
    g1.set(xlabel=None)
    g1.set(ylabel="Frequency")
    g1.set(xticklabels=[])
    plt.tight_layout()
    plt.show()
