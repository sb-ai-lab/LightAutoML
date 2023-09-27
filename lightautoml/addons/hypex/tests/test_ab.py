import pytest
import pandas as pd
from lightautoml.addons.hypex.ABTesting.ab_tester import ABTest


@pytest.fixture
def ab_test():
    return ABTest()


@pytest.fixture
def data():
    return pd.DataFrame(
        {"group": ["test", "test", "control", "control"], "value": [1, 2, 3, 4]}
    )


@pytest.fixture
def target_field():
    return "value"


@pytest.fixture
def group_field():
    return "group"


def test_split_ab(ab_test, data, group_field):
    expected_result = {
        "test": pd.DataFrame({"group": ["test", "test"], "value": [1, 2]}),
        "control": pd.DataFrame({"group": ["control", "control"], "value": [3, 4]}),
    }
    result = ab_test.split_ab(data, group_field)
    assert result == expected_result


def test_calc_difference(ab_test, data, group_field, target_field):
    splitted_data = ab_test.split_ab(data, group_field)
    expected_result = {"ate": -1.0}
    result = ab_test.calc_difference(splitted_data, target_field)
    assert result == expected_result


def test_calc_p_value(ab_test, data, group_field, target_field):
    splitted_data = ab_test.split_ab(data, group_field)
    expected_result = {"t_test": 0.5714285714285714, "mann_whitney": 0.3333333333333333}
    result = ab_test.calc_p_value(splitted_data, target_field)
    assert result == expected_result


def test_execute(ab_test, data, group_field, target_field):
    expected_result = {
        "size": {"test": 2, "control": 2},
        "difference": {"ate": -1.0},
        "p_value": {"t_test": 0.5714285714285714, "mann_whitney": 0.3333333333333333},
    }
    result = ab_test.execute(data, target_field, group_field)
    assert result == expected_result
