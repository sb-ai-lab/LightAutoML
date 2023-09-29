from lightautoml.addons.hypex.ABTesting.ab_tester import ABTest
from lightautoml.addons.hypex.utils.tutorial_data_creation import create_test_data


# def test_split_ab():
#     data = create_test_data()
#     half_data = int(data.shape[0] / 2)
#     data['group'] = ['test'] * half_data + ['control'] * half_data
#
#     group_field = 'group'
#
#     model = ABTest()
#     splitted_data = model.split_ab(data, group_field)
#
#     assert isinstance(splitted_data, dict), "result of split_ab is not dict"
#     assert len(splitted_data) == 2, "split_ab contains not of 2 values"
#     assert list(splitted_data.keys()) == ['test', 'control'], "changed keys in result of split_ab"
#
#
# def test_calc_difference():
#     data = create_test_data()
#     half_data = int(data.shape[0] / 2)
#     data['group'] = ['test'] * half_data + ['control'] * half_data
#
#     group_field = 'group'
#     target_field = 'post_spends'
#
#     model = ABTest()
#     splitted_data = model.split_ab(data, group_field)
#     differences = model.calc_difference(splitted_data, target_field)
#
#     assert isinstance(differences, dict), "result of calc_difference is not dict"


def test_calc_p_value():
    data = create_test_data()
    half_data = int(data.shape[0] / 2)
    data['group'] = ['test'] * half_data + ['control'] * half_data

    group_field = 'group'
    target_field = 'post_spends'

    model = ABTest()
    splitted_data = model.split_ab(data, group_field)
    pvalues = model.calc_p_value(splitted_data, target_field)

    assert isinstance(pvalues, dict), "result of calc_p_value is not dict"


def test_execute():
    data = create_test_data()
    half_data = int(data.shape[0] / 2)
    data['group'] = ['test'] * half_data + ['control'] * half_data

    target_field = 'post_spends'
    target_field_before = 'pre_spends'
    group_field = 'group'

    model = ABTest()
    result = model.execute(
        data=data,
        target_field=target_field,
        target_field_before=target_field_before,
        group_field=group_field
    )

    assert isinstance(result, dict), "result of func execution is not dict"
    assert len(result) == 3, "result of execution is changed, len of dict was 3"
    assert list(result.keys()) == ['size', 'difference', 'p_value']
