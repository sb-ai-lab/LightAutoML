from lightautoml.addons.hypex.utils.tutorial_data_creation import create_test_data


def test_aa():
    data = create_test_data(nans_periods=10)
    info_col = 'user_id'
    model = AATest(data=data, target_fields=['pre_spends', 'post_spends'])
