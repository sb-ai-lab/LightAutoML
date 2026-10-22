"""Quantile binning on small, large, and constant training samples."""

import numpy as np
import pytest
import torch

from lightautoml.text.embed import PiecewiseLinearEmbeddings
from lightautoml.transformers.categorical import compute_bins


@pytest.mark.parametrize("n_rows", [2, 3, 20, 48, 49])
def test_default_bins_on_small_samples(n_rows):
    data = torch.arange(n_rows * 2, dtype=torch.float32).reshape(n_rows, 2)
    bins = compute_bins(data)
    assert len(bins) == 2
    for i, edges in enumerate(bins):
        assert edges[0] == data[:, i].min()
        assert edges[-1] == data[:, i].max()
        assert (edges[1:] > edges[:-1]).all()


def test_exact_quantiles_are_preserved_without_subsampling():
    data = torch.tensor(np.random.RandomState(42).normal(size=(100, 4)), dtype=torch.float32)
    expected = torch.quantile(data, torch.linspace(0, 1, 49), dim=0).T
    for actual, wanted in zip(compute_bins(data), expected):
        torch.testing.assert_close(actual, wanted.unique())


def test_subsampling_is_bounded_repeatable_and_preserves_extremes(monkeypatch):
    data = torch.arange(2000, dtype=torch.float32).reshape(1000, 2)
    quantile = torch.quantile
    sizes = []

    def limited_quantile(values, *args, **kwargs):
        sizes.append(values.numel())
        assert values.ndim == 1
        assert values.numel() <= 20
        return quantile(values, *args, **kwargs)

    monkeypatch.setattr(torch, "quantile", limited_quantile)
    first = compute_bins(data, subsample=20)
    np.random.seed(11)
    torch.manual_seed(17)
    second = compute_bins(data, subsample=20)
    assert sizes == [20] * 4
    for i, (left, right) in enumerate(zip(first, second)):
        torch.testing.assert_close(left, right)
        assert left[0] == data[:, i].min()
        assert left[-1] == data[:, i].max()


@pytest.mark.parametrize("value", [0.0, -3.0, 5.0, 1e30, -1e30])
def test_constant_fold_has_finite_piecewise_embeddings(value):
    data = torch.full((20, 1), value)
    bins = compute_bins(data)
    assert len(bins[0]) == 2
    embedder = PiecewiseLinearEmbeddings(num_dims=1, bins=bins, embedding_size=4)
    assert torch.isfinite(embedder({"cont": data})).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_values_are_rejected_even_outside_subsample(value):
    data = torch.ones(1000, 1)
    data[999] = value
    with pytest.raises(ValueError, match="nan/inf"):
        compute_bins(data, subsample=2)


@pytest.mark.parametrize("n_bins", [0, 1, 2.5])
def test_invalid_bin_counts(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        compute_bins(torch.ones(20, 1), n_bins=n_bins)


def test_tree_bins_on_small_sample():
    data = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    bins = compute_bins(data, tree_kwargs={"random_state": 42}, y=data[:, 0], regression=True)
    assert len(bins) == 2
    assert all(torch.isfinite(edges).all() for edges in bins)
