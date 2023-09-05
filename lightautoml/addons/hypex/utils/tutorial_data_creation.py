import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path('.').absolute().parents[0]
sys.path.append(str(ROOT))


def create_test_data(num_users: int = 10000, file_name: str = None):
    # Simulating dataset with known effect size
    num_months = 12

    # signup_months == 0 means customer did not sign up
    signup_months = np.random.choice(np.arange(1, num_months), num_users) * np.random.randint(0, 2, size=num_users)

    df = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(num_users), num_months),
            "signup_month": np.repeat(signup_months, num_months),  # signup month == 0 means customer did not sign up
            "month": np.tile(np.arange(1, num_months + 1), num_users),  # months are from 1 to 12
            "spend": np.random.poisson(500, num_users * num_months),
        }
    )

    # A customer is in the treatment group if and only if they signed up
    df["treat"] = df["signup_month"] > 0

    # Simulating an effect of month (monotonically decreasing--customers buy less later in the year)
    df["spend"] = df["spend"] - df["month"] * 10

    # Simulating a simple treatment effect of 100
    after_signup = (df["signup_month"] < df["month"]) & (df["treat"])
    df.loc[after_signup, "spend"] = df[after_signup]["spend"] + 100

    # Setting the signup month (for ease of analysis)
    i = 3
    df_i_signupmonth = (
        df[df.signup_month.isin([0, i])]
        .groupby(["user_id", "signup_month", "treat"])
        .apply(
            lambda x: pd.Series(
                {"pre_spends": x.loc[x.month < i, "spend"].mean(), "post_spends": x.loc[x.month > i, "spend"].mean(),}
            )
        )
        .reset_index()
    )

    # Additional category features
    gender_i = np.random.choice(a=[0, 1], size=df_i_signupmonth.user_id.nunique())
    gender = [["M", "F"][i] for i in gender_i]

    age = np.random.choice(a=range(18, 70), size=df_i_signupmonth.user_id.nunique())

    industry_i = np.random.choice(a=range(1, 3), size=df_i_signupmonth.user_id.nunique())
    industry_names = ["Finance", "E-commerce", "Logistics"]
    industry = [industry_names[i] for i in industry_i]

    df_i_signupmonth["age"] = age
    df_i_signupmonth["gender"] = gender
    df_i_signupmonth["industry"] = industry
    df_i_signupmonth["industry"] = df_i_signupmonth["industry"].astype("str")
    df_i_signupmonth["treat"] = df_i_signupmonth["treat"].astype(int)

    if file_name is not None:
        df_i_signupmonth.to_csv(ROOT / f"{file_name}.csv", index=False)

    return df_i_signupmonth


create_test_data(num_users=10_000, file_name="Tutorial_data")