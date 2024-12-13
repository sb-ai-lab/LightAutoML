import calendar
import logging

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import pandas as pd
import seaborn as sns
from ...dataset.utils import concatenate


logger = logging.getLogger(__name__)


def calc_one_feat_imp(iters, level, feat, model, data, norm_score, target, metric, silent):
    # data is a LAMLDataset
    initial_col = copy(data[:, feat].data)
    new_col = np.random.permutation(initial_col)
    data[feat] = new_col
    dataset = copy(data)

    # iterate through higher levels
    for n, level in enumerate(model.levels[level:], level + 1):
        level_predictions = []
        for _n, ml_pipe in enumerate(level):
            level_predictions.append(ml_pipe.predict(dataset))

        if n != len(model.levels):

            level_predictions = concatenate(level_predictions)

            if model.skip_conn:

                try:
                    # convert to initial dataset type
                    level_predictions = dataset.from_dataset(level_predictions)
                except TypeError:
                    raise TypeError("Can not convert prediction dataset type to input features. Set skip_conn=False")
                dataset = concatenate([level_predictions, dataset])
            else:
                dataset = level_predictions

    preds = model.blender.predict(level_predictions)
    data[feat] = initial_col
    preds.target = target.values
    new_score = metric(preds)

    if not silent:
        logger.info3("{}/{} Calculated score for {}: {:.7f}".format(iters[0], iters[1], feat, norm_score - new_score))
    return feat, norm_score - new_score


def calc_feats_permutation_imps(model, used_feats, data, target, metric, silent=False):
    n_used_feats = len(used_feats)
    if not silent:
        logger.info3("LightAutoML ts master used {} feats".format(n_used_feats))
    data = data.reset_index(drop=True)
    preds = model.predict(data)
    target_col = data[target]
    preds.target = target_col.values
    norm_score = metric(preds)
    feat_imp = []

    # construct levels of used feats
    used_feats_leveled = {}

    # input data features
    initial_feats = data.drop(columns=target).columns.values
    used_feats_leveled[0] = initial_feats

    # construct levels of stacking feats
    # only consider features not in initial_feats
    for feat in list(set(used_feats) - set(initial_feats)):
        # stacking feat looks like 'Lvl_0_Pipe_1_Mod_2_CatBoost_prediction_1'
        level = feat.split("_")[1]
        level = int(level) + 1

        arr = used_feats_leveled.get(level, [])
        arr.append(feat)
        used_feats_leveled[level] = arr

    # convert holdout data to LAMLDataset
    data = model.reader.read(data, add_array_attrs=False)
    used_feats_leveled[0] = [feature for feature in data.features if feature not in target]

    # iterate through all the levels
    for level in sorted(used_feats_leveled.keys()):
        # compute importances for features
        for it, f in enumerate(used_feats_leveled[level]):
            feat_imp.append(
                calc_one_feat_imp(
                    (it + 1, n_used_feats),
                    level,
                    f,
                    model,
                    data,
                    norm_score,
                    target_col,
                    metric,
                    silent,
                )
            )
        # update holdout data with stacking features
        if level != len(model.levels) - 1:
            level_predictions = []
            for _n, ml_pipe in enumerate(model.levels[level]):
                level_predictions.append(ml_pipe.predict(data))

            level_predictions = concatenate(level_predictions)
            if model.skip_conn:
                try:
                    # convert to initial dataset type
                    level_predictions = data.from_dataset(level_predictions)
                except TypeError:
                    raise TypeError("Can not convert prediction dataset type to input features. Set skip_conn=False")
                data = concatenate([level_predictions, data])
            else:
                data = level_predictions

    feat_imp = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
    feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    return feat_imp


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
                        "y": ys[i].ravel(),
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
        if reader.targets_mapping:
            classes = sorted(reader.targets_mapping, key=reader.targets_mapping.get)[:top_n_classes]
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
