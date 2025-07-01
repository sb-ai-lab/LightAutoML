#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split

from lightautoml.dataset.np_pd_dataset import NumpyDataset

# from lightautoml.dataset.np_pd_dataset import *
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import FoldsRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.roles import TargetRole
from lightautoml.tasks import Task


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def get_target_name(roles):
    for key, value in roles.items():
        if (key == "target") or isinstance(key, TargetRole):
            return value


@pytest.fixture()
def jobs_train_test(nrows=None):
    data = pd.read_csv("./examples/data/jobs_train.csv", nrows=nrows)
    train_data, test_data = train_test_split(
        data.drop("enrollee_id", axis=1), test_size=0.2, stratify=data["target"], random_state=RANDOM_STATE
    )

    return train_data, test_data


@pytest.fixture()
def jobs_roles():
    return {"target": "target"}


@pytest.fixture()
def uplift_data_train_test(sampled_app_roles, nrows=None):
    data = pd.read_csv(
        "./examples/data/sampled_app_train.csv",
        nrows=nrows,
    )
    sampled_app_roles["treatment"] = "CODE_GENDER"

    data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(
        str
    )
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)
    data["report_dt"] = np.datetime64("2018-01-01")
    data["constant"] = 1
    data["allnan"] = np.nan
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)
    data["CODE_GENDER"] = (data["CODE_GENDER"] == "M").astype(int)
    data["__fold__"] = np.random.randint(0, 5, len(data))

    stratify_value = data[get_target_name(sampled_app_roles)] + 10 * data[sampled_app_roles["treatment"]]
    train, test = train_test_split(data, test_size=3000, stratify=stratify_value, random_state=42)
    test_target, test_treatment = (
        test[get_target_name(sampled_app_roles)].values.ravel(),
        test[sampled_app_roles["treatment"]].values.ravel(),
    )

    return train, test, test_target, test_treatment


# @pytest.fixture()
# def uplift_data_roles():
#     return {
#         'target': 'conversion',
#         'treatment': 'CODE_GENDER',
#         CategoryRole(): ['zip_code', 'channel', 'offer'],
#         NumericRole(): ['recency', 'history', 'used_discount', 'used_bogo'],
#     }


@pytest.fixture()
def avito1k_train_test(nrows=None):
    data = pd.read_csv("./examples/data/avito1k_train.csv")
    train_data, test_data = train_test_split(data, test_size=500, random_state=RANDOM_STATE)

    return train_data, test_data


@pytest.fixture()
def avito1k_roles():
    return {
        "target": "deal_probability",
        "group": ["user_id"],
        "text": ["description", "title", "param_1", "param_2", "param_3"],
    }


@pytest.fixture()
def sampled_app_train_test(nrows=None):
    data = pd.read_csv(
        "./examples/data/sampled_app_train.csv",
        nrows=nrows,
    )

    data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(
        np.dtype("timedelta64[D]")
    )
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED", "SK_ID_CURR"], axis=1, inplace=True)

    data["__fold__"] = np.random.randint(0, 5, len(data))

    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=RANDOM_STATE)

    return train_data, test_data


@pytest.fixture()
def sampled_app_roles():
    return {
        TargetRole(): "TARGET",
        CategoryRole(dtype=str): ["NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE"],
        NumericRole(np.float32): ["AMT_CREDIT", "AMT_GOODS_PRICE"],
        DatetimeRole(seasonality=["y", "m", "wd"]): ["BIRTH_DATE", "EMP_DATE"],
        FoldsRole(): "__fold__",
    }


@pytest.fixture()
def ai92_value_77_train_test():
    data = pd.read_csv(
        "./examples/data/ai92_value_77.csv",
    )

    horizon = 30

    train = data[:-horizon]
    test = data[-horizon:]
    return train, test, horizon


@pytest.fixture()
def binary_task():
    return Task("binary")


@pytest.fixture()
def multiclass_task():
    return Task("multiclass")


@pytest.fixture()
def regression_task():
    return Task("reg")


@pytest.fixture()
def lamldataset_with_na():
    return NumpyDataset(
        data=np.array([[1, 2, np.nan], [4, np.nan, np.nan], [7, 8, np.nan]]),
        features=["column0", "column1", "column2"],
        roles={
            "column0": NumericRole(np.float32),
            "column1": NumericRole(np.float32),
            "column2": NumericRole(np.float32)
            # 'target': TargetRole()
        },
        task=Task("binary"),
    )


@pytest.fixture()
def lamldataset_30_2():
    return NumpyDataset(
        data=np.array(
            [
                [-0.13824745, 0.00480088],
                [0.07343245, 0.13640858],
                [0.09652554, 0.14499552],
                [0.18680116, 0.20484195],
                [0.23786176, 0.25568053],
                [0.27613336, 0.28647607],
                [0.27805356, 0.31445874],
                [0.34141948, 0.39048142],
                [0.37229872, 0.40931471],
                [0.37258695, 0.42442431],
                [0.4031683, 0.44681493],
                [0.41302196, 0.44871043],
                [0.47419529, 0.45320404],
                [0.49295444, 0.4621607],
                [0.51143963, 0.53041875],
                [0.51662931, 0.53908724],
                [0.53601089, 0.57561797],
                [0.53873686, 0.58341858],
                [0.57826693, 0.59454063],
                [0.61096581, 0.59672562],
                [0.69025943, 0.6000393],
                [0.71610905, 0.60264963],
                [0.7375221, 0.60708297],
                [0.7446845, 0.66340465],
                [0.80757267, 0.69437259],
                [0.87351977, 0.80059496],
                [0.8831948, 0.86356838],
                [0.94101309, 0.86733969],
                [0.9668895, 0.98769385],
                [1.06743866, 1.0602233],
            ]
        ),
        features=["column0", "column1"],
        roles={
            "column0": NumericRole(np.float32),
            "column1": NumericRole(np.float32),
        },
        task=Task("binary"),
    )
