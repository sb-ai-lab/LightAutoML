import pytest
import torch

from torch.nn import functional as F

from lightautoml.ml_algo.torch_based import _attention
from lightautoml.ml_algo.torch_based.autoint.autoint_utils import AttnInteractionLayer
from lightautoml.ml_algo.torch_based.fttransformer.fttransformer_utils import Attention


class AttentionWeightsSum(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)[1].sum(-1)


def assert_outputs_and_gradients(layer, x, actual, expected):
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    inputs = (x, *layer.parameters())
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated")
def test_scripted_ft_attention_still_returns_weights():
    layer = Attention(dim=8, heads=2, dim_head=4).eval()
    x = torch.randn(3, 5, 8)
    consumer = AttentionWeightsSum(layer)
    torch.testing.assert_close(torch.jit.script(consumer)(x), consumer(x))


@pytest.mark.parametrize("block_fields", [2, 5])
@pytest.mark.parametrize("legacy", [True, False])
def test_ft_attention_weights_and_output_only(monkeypatch, block_fields, legacy):
    torch.manual_seed(42)
    layer = Attention(dim=8, heads=2, dim_head=4, dropout=0.4).double().eval()
    x = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(_attention, "_SCORE_BLOCK_BYTES", block_fields * 3 * 2 * 5 * 8)
    if legacy:
        monkeypatch.delattr(F, "scaled_dot_product_attention", raising=False)
        monkeypatch.setattr(_attention, "_HAS_NON_REENTRANT_CHECKPOINT", False)

    expected, weights = layer(x)
    actual, omitted = layer(x, return_attn=False)
    assert omitted.numel() == 0
    assert weights.shape == (3, 2, 5, 5)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
    assert_outputs_and_gradients(layer, x, actual, expected)


def test_query_block_limit_in_backward(monkeypatch):
    query = torch.randn(3, 5, 2, 4, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    monkeypatch.setattr(_attention, "_SCORE_BLOCK_BYTES", 3 * 2 * 5 * 4 * 2)
    block_sizes = []

    def attend(q, k):
        block_sizes.append(q.size(1))
        return torch.einsum("bqhd,bkhd->bqhk", q, k).softmax(-1)

    out = _attention.attention_by_query(attend, query, key, num_heads=2)
    assert block_sizes == [2, 2, 1]
    out.backward(torch.randn_like(out))
    if _attention._HAS_NON_REENTRANT_CHECKPOINT:
        assert sorted(block_sizes[3:]) == [1, 2, 2]


def test_ft_attention_training_dropout(monkeypatch):
    layer = Attention(dim=8, heads=2, dim_head=4, dropout=1.0).train()
    x = torch.randn(3, 5, 8, requires_grad=True)
    monkeypatch.setattr(_attention, "_SCORE_BLOCK_BYTES", 2 * 3 * 2 * 5 * 4)
    expected, _ = layer(x)
    actual, _ = layer(x, return_attn=False)
    assert torch.count_nonzero(actual) == 0
    assert_outputs_and_gradients(layer, x, actual, expected)


@pytest.mark.parametrize("block_fields,dropout", [(5, 0.35), (2, 0.35), (2, 0.0)])
def test_autoint_preserves_query_value_weighting(monkeypatch, block_fields, dropout):
    torch.manual_seed(42)
    layer = AttnInteractionLayer(3, field_output_size=4, num_heads=2, dropout=dropout).double()
    x = torch.randn(3, 5, 3, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(_attention, "_SCORE_BLOCK_BYTES", block_fields * 3 * 2 * 5 * 8)

    # Original contraction, with the same query blocks so dropout draws the same masks.
    query, key, value = (torch.tensordot(x, w, dims=([-1], [0])) for w in (layer.W_q, layer.W_k, layer.W_v))
    torch.manual_seed(7)
    blocks = []
    for q, v in zip(query.split(block_fields, dim=1), value.split(block_fields, dim=1)):
        scores = torch.einsum("rdoh,rfoh->rdfh", q, key)
        weights = layer.dropout(scores.softmax(dim=2))
        blocks.append(torch.einsum("rfdh,rfoh->rfoh", weights, v))
    expected = torch.cat(blocks, dim=1).flatten(2) + x @ layer.W_r
    expected = layer.layer_norm(F.leaky_relu(expected))
    torch.manual_seed(7)
    actual = layer(x)
    assert_outputs_and_gradients(layer, x, actual, expected)
