import warnings

import torch

from lightautoml.ml_algo.torch_based.node_nn_model import Entmax15Function
from lightautoml.ml_algo.torch_based.node_nn_model import Entmoid15Optimized


def test_entmoid_matches_entmax_forward_and_backward():
    actual_input = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)

    actual = Entmoid15Optimized.apply(actual_input)
    expected = Entmax15Function.apply(torch.stack([expected_input, torch.zeros_like(expected_input)], dim=1), 1)[:, 0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)
        actual.sum().backward()
        expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
