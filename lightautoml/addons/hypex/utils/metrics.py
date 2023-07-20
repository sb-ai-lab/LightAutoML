"""Calculate metrics."""
import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..utils.psi_pandas import report

logger = logging.getLogger("metrics")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


def smd(orig: pd.DataFrame, matched: pd.DataFrame, silent=False) -> pd.DataFrame:
    """Calculates the standardised mean difference to evaluate matching quality.

    Args:
        orig: pd.DataFrame
            Initial dataframe
        matched: pd.DataFrame
            Matched dataframe
        silent: bool, optional
            If silent, logger in info mode

    Returns:
        pd.DataFrame: The standard mean deviation between initial and matched dataframes
    """
    smd_data = abs(orig.mean(0) - matched.mean(0)) / orig.std(0)

    if silent:
        logger.debug(f"Standardised mean difference:\n{smd_data}")
    else:
        logger.info(f"Standardised mean difference:\n{smd_data}")

    return smd_data


def ks(orig: pd.DataFrame, matched: pd.DataFrame, silent=False) -> dict:
    """Performs a Kolmogorov-Smirnov test to evaluate matching quality per columns.

    Args:
        orig: pd.DataFrame
            Initial dataframe
        matched: pd.DataFrame
            Matched dataframe
         silent: bool, optional
            If silent, logger in info mode


    Returns:
        dict: dict of p-values

    """
    ks_dict = dict()
    matched.columns = orig.columns
    for col in orig.columns:
        ks_pval_1 = ks_2samp(orig[col].values, matched[col].values)[1]
        ks_dict.update({col: ks_pval_1})

    filter_list = list(ks_dict.keys())[:3] + list(ks_dict.keys())[-3:]
    dict_to_show = {key: val for key, val in ks_dict.items() if key in filter_list}

    if silent:
        logger.debug(f"Kolmogorov-Smirnov test to check matching quality: \n{dict_to_show}")
    else:
        logger.info(f"Kolmogorov-Smirnov test to check matching quality: \n{dict_to_show}")

    return ks_dict


def matching_quality(data: pd.DataFrame, treatment: str, features: list, features_psi: list,
                     silent: bool = False) -> tuple:
    """Wraps the functionality for estimating matching quality.

    Args:
        data: pd.DataFrame
            The dataframe of matched data
        treatment: str
            The column determining control and test groups
        features: list
            The list of features, ks-test and smd accept only numeric values
        features_psi: list
            The list of features for calculating Population Stability Index (PSI)
         silent: bool, optional
            If silent, logger in info mode


    Returns:
        tuple: A tuple of dataframes with estimated metrics for matched treated to control and control to treated

    """
    orig_treated = data[data[treatment] == 1][features]
    orig_untreated = data[data[treatment] == 0][features]
    matched_treated = data[data[treatment] == 1][sorted([f + "_matched" for f in features])]
    matched_treated.columns = list(map(lambda x: x.replace("_matched", ""), matched_treated.columns))
    matched_untreated = data[data[treatment] == 0][sorted([f + "_matched" for f in features])]
    matched_untreated.columns = list(map(lambda x: x.replace("_matched", ""), matched_untreated.columns))

    psi_treated = data[data[treatment] == 1][features_psi]
    psi_treated_matched = data[data[treatment] == 1][[f + "_matched" for f in features_psi]]
    psi_treated_matched.columns = [f + "_treated" for f in features_psi]
    psi_treated.columns = [f + "_treated" for f in features_psi]

    psi_untreated = data[data[treatment] == 0][features_psi]
    psi_untreated_matched = data[data[treatment] == 0][[f + "_matched" for f in features_psi]]
    psi_untreated.columns = [f + "_untreated" for f in features_psi]
    psi_untreated_matched.columns = [f + "_untreated" for f in features_psi]

    treated_smd_data = smd(orig_treated, matched_treated, silent)
    untreated_smd_data = smd(orig_untreated, matched_untreated, silent)
    smd_data = pd.concat([treated_smd_data, untreated_smd_data], axis=1)
    smd_data.columns = ["match_control_to_treat", "match_treat_to_control"]

    treated_ks = ks(orig_treated, matched_treated, silent)
    untreated_ks = ks(orig_untreated, matched_untreated, silent)
    ks_dict = {k: [treated_ks[k], untreated_ks[k]] for k in treated_ks.keys()}
    ks_df = pd.DataFrame(data=ks_dict, index=range(2)).T
    ks_df.columns = ["match_control_to_treat", "match_treat_to_control"]

    report_cols = ["column", "anomaly_score", "check_result"]
    report_psi_treated = report(psi_treated, psi_treated_matched, silent=silent)[report_cols]
    report_psi_treated.columns = [col + "_treated" for col in report_cols]
    report_psi_untreated = report(psi_untreated, psi_untreated_matched, silent=silent)[report_cols]
    report_psi_untreated.columns = [col + "_untreated" for col in report_cols]
    report_psi = pd.concat(
        [report_psi_treated.reset_index(drop=True), report_psi_untreated.reset_index(drop=True)], axis=1
    )

    return report_psi, ks_df, smd_data


def check_repeats(index: np.array, silent: bool = False) -> float:
    """Checks the fraction of duplicated indexes in the given array.

     Args:
        index: np.ndarray
            The array of indexes to check for duplicates
        silent: bool, optional
            If silent, logger in info mode

    Returns:
        float:
            The fraction of duplicated index
    """
    unique, counts = np.unique(index, return_counts=True)
    rep_frac = len(unique) / len(index) if len(unique) > 0 else 0

    if silent:
        logger.debug(f"Fraction of duplicated indexes: {rep_frac: .2f}")
    else:
        logger.info(f"Fraction of duplicated indexes: {rep_frac: .2f}")

    return round(rep_frac, 2)
