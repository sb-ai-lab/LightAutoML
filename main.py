import warnings

import pandas as pd

from Matcher.pipelines.Matcher import Matching

warnings.filterwarnings("ignore")
target = "sum_oper"
treatment = "treated"

df = pd.read_csv('Matcher/data/p2p_data_after_MIS.csv').drop(["Unnamed: 0", "report_dt"], axis=1).sample(1000)

print(df)

match = Matching(df, target, treatment)

df_matched, ate = match.estimate()

print(ate)
