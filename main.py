import warnings

import pandas as pd

from matcher.pipelines.matcher import Matching

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    target = "created_variable"
    treatment = "is_tb_pilot"

    df = pd.read_csv('data/data_for_merge_actual.csv').drop(["Unnamed: 0", "cust_inn"], axis=1)

    match = Matching(df, target, treatment, is_feature_select=True)

    df_matched, ate = match.estimate()

    print(ate)
