from lightautoml.addons.hypex.ABTesting.ab_tester import ABTest

import pytest
import pandas as pd
import numpy as np

DATA_SIZE = 100


@pytest.fixture
def ab_test():
    return ABTest()


@pytest.fixture
def data():
    # Generate synthetic data for group A
    group_a_data = np.random.normal(loc=10, scale=2, size=DATA_SIZE)
    # Generate synthetic data for group B
    group_b_data = np.random.normal(loc=12, scale=2, size=DATA_SIZE)
    group_bp_data = np.random.normal(loc=10, scale=2, size=DATA_SIZE * 2)
    return pd.DataFrame(
        {
            "group": ["control"] * len(group_a_data) + ["test"] * len(group_b_data),
            "value": list(group_a_data) + list(group_b_data),
            "previous_value": group_bp_data,
        }
    )


@pytest.fixture
def target_field():
    return "value"


@pytest.fixture
def group_field():
    return "group"


@pytest.fixture
def previous_value():
    return "previous_value"


@pytest.fixture
def alpha():
    return 0.05


def test_split_ab(ab_test, data, group_field):
    result = ab_test.split_ab(data, group_field)
    assert len(result["test"]) == DATA_SIZE
    assert len(result["control"]) == DATA_SIZE


def test_calc_difference(ab_test, data, group_field, target_field, previous_value):
    splitted_data = ab_test.split_ab(data, group_field)
    result = ab_test.calc_difference(splitted_data, target_field, previous_value)
    assert 1 < result["ate"] < 3
    assert 1 < result["cuped"] < 3
    assert 1 < result["diff_in_diff"] < 3


def test_calc_difference_with_previous_value(
    ab_test, data, group_field, target_field, previous_value
):
    ab_test.calc_difference_method = "ate"
    splitted_data = ab_test.split_ab(data, group_field)
    result = ab_test.calc_difference(splitted_data, previous_value)
    assert -1 < result["ate"] < 1


def test_calc_p_value(ab_test, data, group_field, target_field, previous_value, alpha):
    splitted_data = ab_test.split_ab(data, group_field)
    result = ab_test.calc_p_value(splitted_data, target_field)
    assert result["t_test"] < alpha
    assert result["mann_whitney"] < alpha

    result = ab_test.calc_p_value(splitted_data, previous_value)
    assert result["t_test"] > alpha
    assert result["mann_whitney"] > alpha


def test_execute(ab_test, data, group_field, target_field, previous_value, alpha):
    result = ab_test.execute(data, target_field, group_field, previous_value)
    print(result)
    assert result["size"]["test"] == DATA_SIZE
    assert result["size"]["control"] == DATA_SIZE
    assert 1 < result["difference"]["ate"] < 3
    assert 1 < result["difference"]["cuped"] < 3
    assert 1 < result["difference"]["diff_in_diff"] < 3
    assert result["p_value"]["t_test"] < alpha
    assert result["p_value"]["mann_whitney"] < alpha
