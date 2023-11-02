import pandas as pd
import sys
from pathlib import Path

from lightautoml.addons.hypex import Matcher
from lightautoml.addons.hypex.utils.tutorial_data_creation import create_test_data

ROOT = Path(".").absolute().parents[0]
sys.path.append(str(ROOT))


# добавить дату в данные и пофиксить баги с этим
# учесть если info_col передается листом из одного значения или строкой


def create_model(group_col: str = None):
    data = pd.read_csv(ROOT / "Tutorial_data.csv")
    info_col = ["user_id", "signup_month"]
    outcome = "post_spends"
    treatment = "treat"

    model = Matcher(input_data=data, outcome=outcome, treatment=treatment, info_col=info_col, group_col=group_col)

    return model


def test_matcher_pos():
    model = create_model()
    res, quality_res, df_matched = model.estimate()

    assert len(model.quality_result.keys()) == 4, "quality results return not four metrics"
    assert list(model.quality_result.keys()) == ["psi", "ks_test", "smd", "repeats"], "metrics renamed"

    assert list(model.results.index) == ["ATE", "ATC", "ATT"], "format of results is changed: type of effects"
    assert list(model.results.columns) == [
        "effect_size",
        "std_err",
        "p-val",
        "ci_lower",
        "ci_upper",
        "post_spends",
    ], "format of results is changed: columns in report"
    assert model.results["p-val"].values[0] <= 0.05, "p-value on ATE is greater than 0.1"
    assert model.results["p-val"].values[1] <= 0.05, "p-value on ATC is greater than 0.1"
    assert model.results["p-val"].values[2] <= 0.05, "p-value on ATT is greater than 0.1"

    assert isinstance(res, tuple), "result of function estimate is not tuple"
    assert len(res) == 3, "tuple does not return 3 values"


def test_matcher_group_pos():
    model = create_model(group_col="industry")
    res, quality_res, df_matched = model.estimate()

    assert len(model.quality_result.keys()) == 4, "quality results return not 4 metrics"
    assert list(model.quality_result.keys()) == [
        "psi",
        "ks_test",
        "smd",
        "repeats",
    ], "metrics renamed, there should be ['psi', 'ks_test', 'smd', 'repeats']"

    assert list(model.results.index) == [
        "ATE",
        "ATC",
        "ATT",
    ], "format of results is changed: type of effects (ATE, ATC, ATT)"
    assert list(model.results.columns) == [
        "effect_size",
        "std_err",
        "p-val",
        "ci_lower",
        "ci_upper",
        "post_spends",
    ], "format of results is changed: columns in report ['effect_size', 'std_err', 'p-val', 'ci_lower', 'ci_upper']"
    assert model.results["p-val"].values[0] <= 0.05, "p-value on ATE is greater than 0.1"
    assert model.results["p-val"].values[1] <= 0.05, "p-value on ATC is greater than 0.1"
    assert model.results["p-val"].values[2] <= 0.05, "p-value on ATT is greater than 0.1"

    assert isinstance(res, tuple), "result of function estimate is not tuple"
    assert len(res) == 3, "tuple does not return 3 values"


def test_matcher_big_data_pos():
    data = create_test_data(1_000_000)
    info_col = ["user_id", "signup_month"]
    outcome = "post_spends"
    treatment = "treat"

    model = Matcher(input_data=data, outcome=outcome, treatment=treatment, info_col=info_col)
    results, quality_results, df_matched = model.estimate()

    assert isinstance(model.estimate(), tuple), "result of function estimate is not tuple"
    assert len(model.estimate()) == 3, "tuple does not return 3 values"
    assert len(quality_results.keys()) == 4, "quality results return not four metrics"


def test_lama_feature_pos():
    model = create_model()
    res = model.lama_feature_select()

    assert len(res) > 0, "features return empty"


def test_validate_result_pos():
    model = create_model()
    model.estimate()
    res = model.validate_result()
    """
                refuter: str
                Refuter type (`random_treatment` , `random_feature` default, `subset_refuter`)
    """

    assert len(res) > 0, "features return empty"
    assert list(model.pval_dict.values())[0][1] > 0.05, "p-value on validate results is less than 0.05"
