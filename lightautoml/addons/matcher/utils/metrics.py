import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def smd(orig, matched, treatment):
    """Standardized mean difference to check matching quality

    Args:
        orig: pd.Dataframe or Any
        matched: pd.Dataframe or Any
        treatment: pd.Series or Any

    Returns:
        Tuple of smd df and figure

    """
    treated = orig[treatment == 1]
    untreated = orig[treatment == 0]
    treated_matched = matched[treatment == 1]
    untreated_matched = matched[treatment == 0]
    smd_data = pd.concat([abs(treated.mean(0) - treated_matched.mean(0)) / treated.std(0),
                          abs(untreated.mean(0) - untreated_matched.mean(0)) / untreated.std(0)], axis=1)
    smd_data.columns = ['match_control_to_treat', 'match_treat_to_control']
    f, axes = plt.subplots(1, 2, figsize=(15, 7))
    sns.violinplot(data=smd_data, ax=axes[0], cut=0)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
    sns.heatmap(data=smd_data, ax=axes[1])
    return smd_data, f


def ks(orig, matched, treatment):
    """Kolmogorov-Smirnov test to check matching quality by columns

    Args:
        orig: pd.Dataframe or Any
        matched: pd.Dataframe or Any
        treatment: pd.Series or Any

    Returns:
        Checked dataframe

    """
    ks_dict = dict()
    matched.columns = orig.columns
    for col in orig.columns:
        ks_pval_1 = ks_2samp(orig[treatment == 1][col].values, matched[treatment == 1][col].values)[1]
        ks_pval_2 = ks_2samp(orig[treatment == 0][col].values, matched[treatment == 0][col].values)[1]
        ks_dict.update({col: [ks_pval_1, ks_pval_2]})
    ks_df = pd.DataFrame(data=ks_dict, index=range(2)).T
    ks_df.columns = ['match_control_to_treat', 'match_treat_to_control']
    return ks_df
