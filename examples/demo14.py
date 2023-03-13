import pandas as pd

from lightautoml.addons.matcher import Matcher


df = pd.read_csv("data/sampled_matching.csv").drop(["Unnamed: 0"], axis=1)

print(df.shape)
print(df.columns)

target = "created_variable"
treatment = "is_tb_pilot"


matcher = Matcher(df, target, treatment, is_feature_select=False, quality_check=True)

matcher.estimate()

print(matcher.matcher.ATE)
print(matcher.matcher.quality_dict)
