import pandas as pd
import numpy as np

from lightautoml.addons.hypex import Matcher


# import sys
# from pathlib import Path
# ROOT = Path('.').absolute()
# print(ROOT)
# sys.path.append('tests')

# def create_test_data(num_users: int = 10000):
#     # Simulating dataset with known effect size
#     num_months = 12
#
#     # signup_months == 0 means customer did not sign up
#     signup_months = np.random.choice(
#         np.arange(1, num_months), num_users
#     ) * np.random.randint(0, 2, size=num_users)
#
#     df = pd.DataFrame({
#         'user_id': np.repeat(np.arange(num_users), num_months),
#         'signup_month': np.repeat(signup_months, num_months),  # signup month == 0 means customer did not sign up
#         'month': np.tile(np.arange(1, num_months + 1), num_users),  # months are from 1 to 12
#         'spend': np.random.poisson(500, num_users * num_months)
#     })
#
#     # A customer is in the treatment group if and only if they signed up
#     df["treat"] = df["signup_month"] > 0
#
#     # Simulating an effect of month (monotonically decreasing--customers buy less later in the year)
#     df["spend"] = df["spend"] - df["month"] * 10
#
#     # Simulating a simple treatment effect of 100
#     after_signup = (df["signup_month"] < df["month"]) & (df["treat"])
#     df.loc[after_signup, "spend"] = df[after_signup]["spend"] + 100
#
#     # Setting the signup month (for ease of analysis)
#     i = 3
#     df_i_signupmonth = (
#         df[df.signup_month.isin([0, i])]
#             .groupby(["user_id", "signup_month", "treat"])
#             .apply(
#             lambda x: pd.Series(
#                 {
#                     "pre_spends": x.loc[x.month < i, "spend"].mean(),
#                     "post_spends": x.loc[x.month > i, "spend"].mean(),
#                 }
#             )
#         )
#             .reset_index()
#     )
#
#     # Additional category features
#     gender_i = np.random.choice(a=[0, 1], size=df_i_signupmonth.user_id.nunique())
#     gender = [['M', 'F'][i] for i in gender_i]
#
#     age = np.random.choice(a=range(18, 70), size=df_i_signupmonth.user_id.nunique())
#
#     industry_i = np.random.choice(a=range(1, 3), size=df_i_signupmonth.user_id.nunique())
#     industry_names = ['Finance', 'E-commerce', 'Logistics']
#     industry = [industry_names[i] for i in industry_i]
#
#     df_i_signupmonth['age'] = age
#     df_i_signupmonth['gender'] = gender
#     df_i_signupmonth['industry'] = industry
#     df_i_signupmonth['industry'] = df_i_signupmonth['industry'].astype('str')
#     df_i_signupmonth['treat'] = df_i_signupmonth['treat'].astype(int)
#
#     return df_i_signupmonth
#
#
# def create_model(group_col: str = None):
#     data = create_test_data()
#     info_col = ['user_id', 'signup_month']
#     outcome = 'post_spends'
#     treatment = 'treat'
#
#     if group_col is not None:
#         group_col = 'industry'
#         model = Matcher(input_data=data, outcome=outcome, treatment=treatment, info_col=info_col, group_col=group_col)
#     else:
#         model = Matcher(input_data=data, outcome=outcome, treatment=treatment, info_col=info_col)
#
#     return model
#
#
# def test_matcher_pos():
#     model = create_model()
#     res = model.estimate()
#
#     assert len(model.quality_result.keys()) == 4, 'quality results return not four metrics'
#     assert list(model.quality_result.keys()) == ['psi', 'ks_test', 'smd', 'repeats'], 'metrics renamed'
#
#     assert list(model.results.index) == ['ATE', 'ATC', 'ATT'], 'format of results is changed: type of effects'
#     assert list(model.results.columns) == ['effect_size', 'std_err', 'p-val', 'ci_lower', 'ci_upper'], 'format of results is changed: columns in report'
#     assert model.results['p-val'].values[0] <= 0.1, 'p-value on ATE is greater than 0.1'
#     assert model.results['p-val'].values[1] <= 0.1, 'p-value on ATC is greater than 0.1'
#     assert model.results['p-val'].values[2] <= 0.1, 'p-value on ATT is greater than 0.1'
#
#     assert isinstance(res, tuple), 'result of function estimate is not tuple'
#     assert len(res) == 3, 'tuple does not return 3 values'
#
#
# def test_matcher_group_pos():
#     model = create_model(group_col='industry')
#     res = model.estimate()
#
#     assert len(model.quality_result.keys()) == 4, 'quality results return not four metrics'
#     assert list(model.quality_result.keys()) == ['psi', 'ks_test', 'smd', 'repeats'], 'metrics renamed'
#
#     assert list(model.results.index) == ['ATE', 'ATC', 'ATT'], 'format of results is changed: type of effects'
#     assert list(model.results.columns) == ['effect_size', 'std_err', 'p-val', 'ci_lower', 'ci_upper'], 'format of results is changed: columns in report'
#     assert model.results['p-val'].values[0] <= 0.05, 'p-value on ATE is greater than 0.1'
#     assert model.results['p-val'].values[1] <= 0.05, 'p-value on ATC is greater than 0.1'
#     assert model.results['p-val'].values[2] <= 0.05, 'p-value on ATT is greater than 0.1'
#
#     assert isinstance(res, tuple), 'result of function estimate is not tuple'
#     assert len(res) == 3, 'tuple does not return 3 values'
#
#
# # def test_matcher_big_data_pos():
# #     data = create_test_data(1_000_000)
# #     info_col = ['user_id', 'signup_month']
# #     outcome = 'post_spends'
# #     treatment = 'treat'
# #
# #     model = Matcher(input_data=data, outcome=outcome, treatment=treatment, info_col=info_col)
# #     results, quality_results, df_matched = model.estimate()
# #
# #     assert isinstance(model.estimate(), tuple), 'result of function estimate is not tuple'
# #     assert len(model.estimate()) == 3, 'tuple does not return 3 values'
# #     assert len(quality_results.keys()) == 4, 'quality results return not four metrics'
#
#
# def test_lama_feature_pos():
#     model = create_model()
#     res = model.lama_feature_select()
#
#     assert len(res) > 0, "features return empty"
#
#
# def test_validate_result_pos():
#     model = create_model()
#     res = model.validate_result()
#
#     # assert len(res) > 0, "features return empty"
#     # assert list(model.pval_dict.values())[0][1] > 0.05
