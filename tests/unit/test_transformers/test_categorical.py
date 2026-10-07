import numpy as np
import pytest

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.tasks import Task
from lightautoml.transformers.categorical import TargetEncoder


def make_target_encoding_dataset(target, task_name="reg"):
    return NumpyDataset(
        data=np.array([[1], [1], [2], [2]], dtype=np.int32),
        features=["category"],
        roles=CategoryRole(np.int32, label_encoded=True),
        task=Task(task_name),
        target=target,
        folds=np.array([0, 1, 0, 1], dtype=np.int32),
    )


@pytest.mark.parametrize("target_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [1, -1])
def test_target_encoder_preserves_fractional_regression_targets(target_dtype, sign):
    target = sign * np.array([0.1, 0.2, 0.8, 0.9], dtype=target_dtype)
    dataset = make_target_encoding_dataset(target.copy())
    original_folds = dataset.folds.copy()
    encoder = TargetEncoder(alphas=(1.0,))

    output = encoder.fit_transform(dataset)

    np.testing.assert_allclose(output.data[:, 0], sign * np.array([0.375, 0.275, 0.725, 0.625]), rtol=1e-6)
    np.testing.assert_array_equal(dataset.target, target)
    np.testing.assert_array_equal(output.target, target)
    np.testing.assert_array_equal(output.folds, original_folds)
    assert output.shape == (4, 1)
    assert output.features == ["oof__category"]
    assert output.data.dtype == np.float32
    assert output.roles["oof__category"].name == "Numeric"
    assert not output.roles["oof__category"].prob

    new_data = NumpyDataset(
        data=np.array([[2], [1], [0]], dtype=np.int32),
        features=["category"],
        roles=CategoryRole(np.int32, label_encoded=True),
        task=dataset.task,
    )
    transformed = encoder.transform(new_data)

    np.testing.assert_allclose(transformed.data[:, 0], sign * np.array([2.2 / 3, 0.8 / 3, 0.5]), rtol=1e-6)
    assert transformed.features == output.features
    assert transformed.data.dtype == np.float32


def test_target_encoder_selects_smoothing_using_fractional_targets():
    dataset = make_target_encoding_dataset(np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float64))
    encoder = TargetEncoder(alphas=(0.5, 10.0))

    output = encoder.fit_transform(dataset)

    np.testing.assert_allclose(output.data[:, 0], [5.9 / 11, 5.1 / 11, 5.1 / 11, 5.9 / 11], rtol=1e-6)


def test_target_encoder_excludes_held_out_fold_targets():
    dataset = make_target_encoding_dataset(np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64))
    held_out_fold = dataset.folds == 0
    original = TargetEncoder(alphas=(1.0,)).fit_transform(dataset)
    changed_target = dataset.target.copy()
    changed_target[held_out_fold] += 100
    changed_dataset = make_target_encoding_dataset(changed_target)

    changed = TargetEncoder(alphas=(1.0,)).fit_transform(changed_dataset)

    np.testing.assert_allclose(changed.data[held_out_fold], original.data[held_out_fold], rtol=1e-6)
    assert not np.allclose(changed.data[~held_out_fold], original.data[~held_out_fold])


@pytest.mark.parametrize(
    "task_name, target_values, expected_oof, expected_transformed",
    [
        ("reg", [1, 2, 8, 9], [3.75, 2.75, 7.25, 6.25], [8 / 3, 8 / 3, 22 / 3, 22 / 3]),
        ("binary", [0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75], [1 / 6, 1 / 6, 5 / 6, 5 / 6]),
    ],
)
def test_target_encoder_preserves_integer_targets(task_name, target_values, expected_oof, expected_transformed):
    target = np.array(target_values, dtype=np.int32)
    dataset = make_target_encoding_dataset(target.copy(), task_name=task_name)
    encoder = TargetEncoder(alphas=(1.0,))

    output = encoder.fit_transform(dataset)
    transformed = encoder.transform(dataset)

    np.testing.assert_allclose(output.data[:, 0], expected_oof, rtol=1e-6)
    np.testing.assert_allclose(transformed.data[:, 0], expected_transformed, rtol=1e-6)
    np.testing.assert_array_equal(dataset.target, target)
    assert output.data.dtype == np.float32
    assert output.roles["oof__category"].prob == (task_name == "binary")
