import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from ..utils.psi_pandas import *
import logging

logger = logging.getLogger('metrics')
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.DEBUG
)

def smd(orig, matched):
    """Standardised mean difference to check matching quality

    Args:
        orig: pd.Dataframe or Any
        matched: pd.Dataframe or Any

    Returns:
        Tuple of smd df and figure

    """
    smd_data = abs(orig.mean(0) - matched.mean(0)) / orig.std(0)
    logger.info(f'Standardised mean difference: {round(smd_data, 4)}')
    return smd_data



def ks(orig, matched):
    """Kolmogorov-Smirnov test to check matching quality by columns

    Args:
        orig: pd.Dataframe
        matched: pd.Dataframe

    Returns:
        ks_dict - dict of p-values: dict

    """
    logger.info('Applying Kolmogorov-Smirnov test to check matching quality')
    ks_dict = dict()
    matched.columns = orig.columns
    for col in orig.columns:
        ks_pval_1 = ks_2samp(orig[col].values, matched[col].values)[1]
        ks_dict.update({col: ks_pval_1})
    return ks_dict


def matching_quality(data, treatment, features, features_psi):
    """Wrapping function for matching quality estimation.

    Args:
        data - df_matched: pd.DataFram
        treatment -  treatment
        features - feature list, kstest and  smd accept only numeric values

    Returns:
        report_psi, ks_df, smd_data - dictionaries with estimated metrics
        for matched treated to control and control to treated: tuple of dicts

    """
    orig_treated = data[data[treatment] == 1][features]
    orig_untreated = data[data[treatment] == 0][features]
    matched_treated = data[data[treatment] == 1][
            [f + '_matched' for f in features]]
    matched_treated.columns = orig_treated.columns
    matched_untreated = data[data[treatment] == 0][
            [f + '_matched' for f in features]]
    matched_untreated.columns = orig_treated.columns

    psi_treated = data[data[treatment] == 1][features_psi]
    psi_treated_matched = data[data[treatment] == 1][[f + '_matched' for f in features_psi]]
    psi_treated_matched.columns = [f + '_treated' for f in features_psi]
    psi_treated.columns = [f + '_treated' for f in features_psi]
    psi_untreated = data[data[treatment] == 0][features_psi]
    psi_untreated_matched = data[data[treatment] == 0][
            [f + '_matched' for f in features_psi]]
    psi_untreated.columns = [f + '_untreated' for f in features_psi]
    psi_untreated_matched.columns = [f + '_untreated' for f in features_psi]
    treated_smd_data = smd(orig_treated, matched_treated)
    untreated_smd_data = smd(orig_untreated, matched_untreated)
    smd_data = pd.concat([treated_smd_data, untreated_smd_data], axis=1)
    smd_data.columns = ['match_control_to_treat', 'match_treat_to_control']
    treated_ks = ks(orig_treated, matched_treated)
    untreated_ks = ks(orig_untreated, matched_untreated)
    ks_dict = {k: [treated_ks[k], untreated_ks[k]] for k in treated_ks.keys()}
    ks_df = pd.DataFrame(data=ks_dict, index=range(2)).T
    ks_df.columns = ['match_control_to_treat', 'match_treat_to_control']
    report_psi_treated = report(psi_treated, psi_treated_matched)[['column', 'anomaly_score', 'check_result']]
    report_psi_untreated = report(psi_untreated, psi_untreated_matched)[['column', 'anomaly_score', 'check_result']]
    report_psi = pd.concat([report_psi_treated.reset_index(drop=True),report_psi_untreated.reset_index(drop=True)], axis=1)
    return report_psi, ks_df, smd_data

def check_repeats(index):
    """The function checks fraction of duplicated indexes.

     Args:
        index: numpy array.

    Returns:
        Fraction of dupicated index, float.
    """
    unique, counts = np.unique(index, return_counts=True)
    rep_frac = len(unique) / len(index) if len(unique) > 0 else 0
    logger.info(f'Fraction of duplicated indexes: {round(rep_frac, 2)}')
    return round(rep_frac, 2)
