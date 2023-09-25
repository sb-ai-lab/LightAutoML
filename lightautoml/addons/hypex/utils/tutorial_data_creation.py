import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Iterable, Union

ROOT = Path('.').absolute().parents[0]
sys.path.append(str(ROOT))


def set_nans(
        data: pd.DataFrame,
        na_step: Union[Iterable[int], int] = None,
        nan_cols: Union[Iterable[str], str] = None
):
    """Fill some values with NaN/

    Args:
        data: input dataframe
        na_step:
            num or list of nums of period to make NaN (step of range)
            If list - iterates accordingly order of columns
        nan_cols:
            name of one or several columns to fill with NaN
            If list - iterates accordingly order of na_step

    Returns:
        data: dataframe with some NaNs
    """
    if (nan_cols is not None) or (na_step is not None):
        # correct type of columns to iterate

        #  number of nans
        if na_step is None:
            na_step = [10]
            print(f'No na_step specified: set to {na_step}')
        elif not isinstance(na_step, Iterable):
            na_step = [na_step]

        #  columns
        if nan_cols is None:
            nan_cols = list(data.columns)
            print('No nan_cols specified. Setting NaNs applied to all columns')
        elif not isinstance(nan_cols, Iterable):
            nan_cols = [nan_cols]

        # correct length of two lists
        if len(na_step) > len(nan_cols):
            na_step = na_step[:len(nan_cols)]
            print('Length of na_step is bigger than length of columns. Used only first values')
        elif len(na_step) < len(nan_cols):
            na_step = na_step + [na_step[-1]] * (len(nan_cols) - len(na_step))
            print('Length of na_step is less than length of columns. Used last value several times')

        # create list of indexes to fill with na
        nans_indexes = [list(range(i, len(data), period)) for i, period in enumerate(na_step)]

        for i in range(len(nan_cols)):
            try:
                data.loc[nans_indexes[i], nan_cols[i]] = np.nan
            except KeyError:
                print(f'There is no column {nan_cols[i]} in data. No nans in this column will be added.')
    else:
        print('No NaN added')

    return data


def create_test_data(
        num_users: int = 10000,
        na_step: Union[Iterable[int], int] = None,
        nan_cols: Union[Iterable[str], str] = None,
        file_name: str = None
):
    """Creates data for tutorial.

    Args:
        num_users: num of strings
        na_step: 
            num or list of nums of period to make NaN (step of range)
            If list - iterates accordingly order of columns
        nan_cols: 
            name of one or several columns to fill with NaN
            If list - iterates accordingly order of na_step
        file_name: name of file to save; doesn't save file if None

    Returns:
        data: dataframe with
    """
    # Simulating dataset with known effect size
    num_months = 12

    # signup_months == 0 means customer did not sign up
    signup_months = np.random.choice(np.arange(1, num_months), num_users) * np.random.randint(0, 2, size=num_users)

    data = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(num_users), num_months),
            "signup_month": np.repeat(signup_months, num_months),  # signup month == 0 means customer did not sign up
            "month": np.tile(np.arange(1, num_months + 1), num_users),  # months are from 1 to 12
            "spend": np.random.poisson(500, num_users * num_months),
        }
    )

    # A customer is in the treatment group if and only if they signed up
    data["treat"] = data["signup_month"] > 0

    # Simulating an effect of month (monotonically decreasing--customers buy less later in the year)
    data["spend"] = data["spend"] - data["month"] * 10

    # Simulating a simple treatment effect of 100
    after_signup = (data["signup_month"] < data["month"]) & (data["treat"])
    data.loc[after_signup, "spend"] = data[after_signup]["spend"] + 100

    # Setting the signup month (for ease of analysis)
    i = 3
    data = (
        data[data.signup_month.isin([0, i])]
        .groupby(["user_id", "signup_month", "treat"])
        .apply(
            lambda x: pd.Series(
                {"pre_spends": x.loc[x.month < i, "spend"].mean(), "post_spends": x.loc[x.month > i, "spend"].mean(),}
            )
        )
        .reset_index()
    )

    # Additional category features
    gender_i = np.random.choice(a=[0, 1], size=data.user_id.nunique())
    gender = [["M", "F"][i] for i in gender_i]

    age = np.random.choice(a=range(18, 70), size=data.user_id.nunique())

    industry_i = np.random.choice(a=range(1, 3), size=data.user_id.nunique())
    industry_names = ["Finance", "E-commerce", "Logistics"]
    industry = [industry_names[i] for i in industry_i]

    data["age"] = age
    data["gender"] = gender
    data["industry"] = industry
    data["industry"] = data["industry"].astype("str")
    data["treat"] = data["treat"].astype(int)

    # input nans in data if needed
    data = set_nans(data, na_step, nan_cols)

    if file_name is not None:
        data.to_csv(ROOT / f"{file_name}.csv", index=False)

    return data


# create_test_data(num_users=10_000, file_name="Tutorial_data")