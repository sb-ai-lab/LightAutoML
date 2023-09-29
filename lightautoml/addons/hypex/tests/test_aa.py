import pandas as pd
from lightautoml.addons.hypex.ABTesting.ab_tester import AATest
from lightautoml.addons.hypex.utils.tutorial_data_creation import create_test_data


def test_aa_simple():
    data = create_test_data(rs=52)
    info_col = "user_id"
    iterations = 20

    model = AATest(
        data=data,
        target_fields=["pre_spends", "post_spends"],
        info_cols=info_col
    )
    res, datas_dict = model.search_dist_uniform_sampling(iterations=iterations)

    assert isinstance(res, pd.DataFrame), "Metrics are not dataframes"
    assert res.shape[0] == iterations, "Metrics dataframe contains more or less rows with random states " \
                                       "(#rows should be equal #of experiments"
    assert info_col not in model.data, "Info_col is take part in experiment, it should be deleted in preprocess"
    assert isinstance(datas_dict, dict), "Result is not dict"
    assert len(datas_dict) == iterations, "# of dataframes is not equal # of iterations"
    assert all(data.columns) == all(datas_dict[0].drop(columns=['group']).columns), \
        "Columns in the result are not the same as columns in initial data "


def test_aa_group():
    data = create_test_data(rs=52)
    info_col = "user_id"
    group_cols = 'industry'
    iterations = 20

    model = AATest(
        data=data,
        target_fields=["pre_spends", "post_spends"],
        info_cols=info_col,
        group_cols=group_cols
    )
    res, datas_dict = model.search_dist_uniform_sampling(iterations=iterations)

    assert isinstance(res, pd.DataFrame), "Metrics are not dataframes"
    assert res.shape[0] == iterations, "Metrics dataframe contains more or less rows with random states " \
                                       "(#rows should be equal #of experiments"
    assert info_col not in model.data, "Info_col is take part in experiment, it should be deleted in preprocess"
    assert isinstance(datas_dict, dict), "Result is not dict"
    assert len(datas_dict) == iterations, "# of dataframes is not equal # of iterations"
    assert all(data.columns) == all(datas_dict[0].drop(columns=['group']).columns), "Columns in the result are not " \
                                                                                    "the same as columns in initial " \
                                                                                    "data "


def test_aa_quantfields():
    data = create_test_data(rs=52)
    info_col = "user_id"
    group_cols = 'industry'
    quant_field = 'gender'
    iterations = 20

    model = AATest(
        data=data,
        target_fields=["pre_spends", "post_spends"],
        info_cols=info_col,
        group_cols=group_cols,
        quant_field=quant_field
    )
    res, datas_dict = model.search_dist_uniform_sampling(iterations=iterations)

    assert isinstance(res, pd.DataFrame), "Metrics are not dataframes"
    assert res.shape[0] == iterations, "Metrics dataframe contains more or less rows with random states " \
                                       "(#rows should be equal #of experiments"
    assert info_col not in model.data, "Info_col is take part in experiment, it should be deleted in preprocess"
    assert isinstance(datas_dict, dict), "Result is not dict"
    assert len(datas_dict) == iterations, "# of dataframes is not equal # of iterations"
    assert all(data.columns) == all(datas_dict[0].drop(columns=['group']).columns), "Columns in the result are not " \
                                                                                    "the same as columns in initial " \
                                                                                    "data "

