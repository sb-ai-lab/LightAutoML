#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split

# from lightautoml.dataset.np_pd_dataset import *
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import FoldsRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.roles import TargetRole
from lightautoml.tasks import Task


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@pytest.fixture()
def jobs_train_test(nrows=None):
    data = pd.read_csv("./examples/data/jobs_train.csv", nrows=nrows)
    train_data, test_data = train_test_split(data.drop("enrollee_id", axis=1), test_size=0.2, stratify=data["target"])

    return train_data, test_data


@pytest.fixture()
def sampled_app_train_test(nrows=None):
    data = pd.read_csv(
        "./examples/data/sampled_app_train.csv",
        usecols=[
            "TARGET",
            "NAME_CONTRACT_TYPE",
            "AMT_CREDIT",
            "NAME_TYPE_SUITE",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ],
        nrows=nrows,
    )

    data["BIRTH_DATE"] = np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(
        np.dtype("timedelta64[D]")
    )
    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

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
def binary_task():
    return Task("binary")
