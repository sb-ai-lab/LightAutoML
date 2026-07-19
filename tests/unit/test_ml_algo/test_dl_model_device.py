"""Regression tests for device handling of torch-based models (tabnet, autoint, fttransformer).

Covers two bugs:
- inference of a GPU-trained pipeline failed on a CPU-only machine
  (device stored in params was not re-validated at predict time);
- models with sequence input (autoint, fttransformer) could not be trained
  with both numeric and categorical features (wrong concatenation axis
  of the embedder outputs).
"""

import numpy as np
import pytest
import torch

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.ml_algo.dl_model import TorchModel
from lightautoml.tasks import Task
from lightautoml.text.utils import parse_devices
from lightautoml.validation.np_iterators import FoldsIterator


def _make_cat_binary_dataset(n_samples=400, n_num=3, n_cat=2, seed=42):
    np.random.seed(seed)
    num_data = np.random.randn(n_samples, n_num).astype(np.float32)
    cat_data = np.random.randint(0, 4, (n_samples, n_cat)).astype(np.float32)
    data = np.hstack([num_data, cat_data]).astype(np.float32)
    logit = num_data[:, 0] + (cat_data[:, 0] % 2) - 0.5 * num_data[:, 1]
    target = (logit > np.median(logit)).astype(np.int64)
    folds = np.random.randint(0, 2, n_samples)
    features = [f"f_{i}" for i in range(n_num + n_cat)]
    roles = {}
    for i in range(n_num):
        roles[f"f_{i}"] = NumericRole(np.float32)
    for i in range(n_cat):
        roles[f"f_{n_num + i}"] = CategoryRole(np.float32)
    dataset = NumpyDataset(data, features, roles, task=Task("binary"), target=target, folds=folds)
    return dataset


def _make_torch_algo(model_name, device):
    return TorchModel(
        default_params={
            "model": model_name,
            "n_epochs": 2,
            "bs": 64,
            "cat_embedder": "cat",
            "cont_embedder": "cont",
            "embedding_size": 10,
            "device": device,
            "path_to_save": None,
        },
        freeze_defaults=True,
    )


class TestParseDevices:
    def test_string_cuda_device_converted_to_torch_device(self, monkeypatch):
        """String devices like 'cuda:0' must be converted to torch.device."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        device, ids = parse_devices("cuda:0")
        assert isinstance(device, torch.device)
        assert device.type == "cuda" and device.index == 0
        assert ids is None

        device, ids = parse_devices("cuda")
        assert isinstance(device, torch.device)
        assert device.type == "cuda"

    def test_cpu_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        device, ids = parse_devices("cpu")
        assert device == torch.device("cpu")
        assert ids is None

    def test_fallback_to_cpu_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device, ids = parse_devices("cuda:0")
        assert device == torch.device("cpu")
        assert ids is None


@pytest.mark.parametrize("model_name", ["tabnet", "autoint", "fttransformer"])
class TestTorchModelDevice:
    def test_cpu_inference_after_gpu_training(self, model_name, monkeypatch):
        """Predict must work on a CPU-only machine for a pipeline trained with a CUDA device."""
        dataset = _make_cat_binary_dataset()

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        algo = _make_torch_algo(model_name, device)
        algo.fit_predict(FoldsIterator(dataset))

        # emulate a CPU-only inference machine for a GPU-trained pipeline
        algo.params["device"] = torch.device("cuda:0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        pred = algo.predict(dataset).data

        assert pred.shape == (len(dataset), 1)
        assert np.isfinite(pred).all()
        # device must be re-validated at predict time
        assert algo.params["device"] == torch.device("cpu")
        assert algo.params["device_ids"] is None

    def test_fit_with_numeric_and_categorical_features(self, model_name):
        """Models must train with both numeric and categorical features."""
        dataset = _make_cat_binary_dataset()

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        algo = _make_torch_algo(model_name, device)
        oof = algo.fit_predict(FoldsIterator(dataset)).data

        assert oof.shape == (len(dataset), 1)
        assert np.isfinite(oof[~np.isnan(oof)]).all()
