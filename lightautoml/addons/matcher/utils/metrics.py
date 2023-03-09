import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def smd(orig, matched):
    '''Standardized mean difference для проверки качества мэтчинга'''
    smd_data = abs(orig.mean(0) - matched.mean(0)) / orig.std(0)
    return smd_data


def ks(orig, matched):
    '''Тест Колмогорова-Смирнова для поколоночной проверки качества мэтчинга'''
    ks_dict = dict()
    matched.columns = orig.columns
    for col in orig.columns:
        ks_pval_1 = ks_2samp(orig[col].values, matched[col].values)[1]
        ks_dict.update({col: ks_pval_1})
    return ks_dict


