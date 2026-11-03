"""Conditional binning and backward-compatible neural-network checkpoints."""

import pickle

import numpy as np
import pytest
import torch

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.dl_model import TorchModel
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import FoldsIterator


def _dataset():
    return NumpyDataset(
        np.arange(40, dtype=np.float64).reshape(20, 2),
        features=["a", "b"],
        roles=NumericRole(np.float64),
        task=Task("reg"),
        target=np.arange(20, dtype=np.float32),
        folds=np.repeat([0, 1], 10),
    )


def _model(embedder, path=None):
    return TorchModel(
        default_params={
            "model": "mlp",
            "cont_embedder": embedder,
            "n_epochs": 1,
            "bs": 64,
            "num_workers": 0,
            "device": "cpu",
            "path_to_save": path,
            "hidden_size": [8],
            "embedding_size": 4,
        },
        freeze_defaults=True,
    )


def test_plain_embeddings_do_not_compute_bins(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An embedder without piecewise encoding must not compute bins")

    monkeypatch.setattr("lightautoml.ml_algo.dl_model.compute_bins", unexpected)
    dataset = _dataset()
    model = _model("cont")
    predictions = model.fit_predict(FoldsIterator(dataset)).data
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("on_disk", [False, True])
def test_piecewise_bins_keep_legacy_population_dtype_and_checkpoint_layout(tmp_path, on_disk, monkeypatch):
    dataset = _dataset()
    model = _model("piecewise", str(tmp_path) if on_disk else None)
    from lightautoml.ml_algo.dl_model import compute_bins

    calls = []

    def record_bins(data):
        calls.append(data.clone())
        return compute_bins(data)

    monkeypatch.setattr("lightautoml.ml_algo.dl_model.compute_bins", record_bins)
    oof = model.fit_predict(FoldsIterator(dataset)).data
    assert np.isfinite(oof).all()
    assert len(calls) == 1
    assert calls[0].dtype == torch.float32
    assert len(calls[0]) == len(dataset)
    checkpoints = [torch.load(path, weights_only=False) for path in model.models] if on_disk else model.models
    assert all("bins" not in checkpoint for checkpoint in checkpoints)
    for i, bins in enumerate(model.params["bins"]):
        assert bins.dtype == torch.float32
        assert bins[0] == dataset.data[:, i].min()
        assert bins[-1] == dataset.data[:, i].max()
    first = model.predict(dataset).data
    second = model.predict(dataset).data
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()


def test_pickled_piecewise_model_keeps_legacy_bin_parameters():
    dataset = _dataset()
    model = _model("piecewise")
    model.fit_predict(FoldsIterator(dataset, n_folds=1))
    expected = model.predict(dataset).data
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(dataset).data, expected)
