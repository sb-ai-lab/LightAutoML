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
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.dl_model import TorchModel
from lightautoml.ml_algo.torch_based.nn_models import FTTransformer
from lightautoml.ml_algo.torch_based.nn_models import GaussianNoise
from lightautoml.ml_algo.torch_based.nn_models import UniformNoise
from lightautoml.tasks import Task
from lightautoml.text.embed import WeightedCatEmbedding
from lightautoml.text.trainer import Trainer
from lightautoml.text.utils import parse_devices
from lightautoml.validation.np_iterators import FoldsIterator


def _make_cat_binary_dataset(n_samples=400, n_num=3, n_cat=2, seed=42):
    rng = np.random.RandomState(seed)
    num_data = rng.randn(n_samples, n_num).astype(np.float32)
    cat_data = rng.randint(0, 4, (n_samples, n_cat)).astype(np.float32)
    data = np.hstack([num_data, cat_data]).astype(np.float32)
    logit = num_data[:, 0] + (cat_data[:, 0] % 2) - 0.5 * num_data[:, 1]
    target = (logit > np.median(logit)).astype(np.int64)
    folds = rng.randint(0, 2, n_samples)
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
    def test_cpu_inference_with_saved_cuda_configuration(self, model_name, monkeypatch):
        """Exercise a serialized CUDA configuration without requiring GPU hardware."""
        dataset = _make_cat_binary_dataset()

        device = torch.device("cpu")
        algo = _make_torch_algo(model_name, device)
        algo.fit_predict(FoldsIterator(dataset))

        # emulate a CPU-only inference machine for a GPU-trained pipeline
        algo.params["device"] = torch.device("cuda:0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        pred = algo.predict(dataset).data

        assert pred.shape == (len(dataset), 1)
        assert np.isfinite(pred).all()
        np.testing.assert_allclose(algo.predict(dataset).data, pred)
        # device must be re-validated at predict time
        assert algo.params["device"] == torch.device("cpu")
        assert algo.params["device_ids"] is None

    def test_fit_with_numeric_and_categorical_features(self, model_name):
        """Models must train with both numeric and categorical features."""
        dataset = _make_cat_binary_dataset()

        device = torch.device("cpu")
        algo = _make_torch_algo(model_name, device)
        oof = algo.fit_predict(FoldsIterator(dataset)).data

        assert oof.shape == (len(dataset), 1)
        assert np.isfinite(oof).all()


@pytest.mark.parametrize("device", [0, "0", "cuda", "cuda:0", torch.device("cuda:0")])
def test_parse_cuda_representations(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    parsed, ids = parse_devices(device)
    assert isinstance(parsed, torch.device)
    assert parsed.type == "cuda"
    assert ids is None


def test_parse_multiple_cuda_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, ids = parse_devices(["cuda:1", "0"], is_dp=True)
    assert device == torch.device("cuda:1")
    assert ids == [1, 0]


@pytest.mark.parametrize("model_type", [FTTransformer, GaussianNoise, UniformNoise])
def test_forward_uses_input_device_after_module_transfer(model_type):
    if model_type is FTTransformer:
        model = model_type(device="cuda:3", embedding_size=4, depth=1, num_enc_layers=1)
        data = torch.ones(3, 2, 4)
    else:
        model = model_type(stddev=0.1, device="cuda:3")
        data = torch.ones(3, 4)
    model.to("cpu")
    output = model(data)
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


def test_weighted_embedding_uses_input_device():
    model = WeightedCatEmbedding(cat_dims=[3], cat_vc=[{0: 1, 1: 2, 2: 2}], device="cuda:3")
    output = model.to("cpu")({"cat": torch.tensor([[0], [1], [2]])})
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


def test_trainer_rebinds_constructor_devices_and_clears_data_parallel(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    dataset = _make_cat_binary_dataset(n_samples=160)
    algo = _make_torch_algo("autoint", "cpu")
    algo.fit_predict(FoldsIterator(dataset))
    trainer = algo._infer_params()
    trainer.device = torch.device("cuda:3")
    trainer.device_ids = [0, 3]
    trainer.net_params["device"] = "cuda:3"
    trainer.net_params["cont_params"]["device"] = "cuda:3"
    trainer.net_params["cat_params"]["device"] = "cuda:3"
    trainer.load_state(algo.models[0])
    assert trainer.device == torch.device("cpu")
    assert trainer.device_ids is None
    assert trainer.net_params["device"] == torch.device("cpu")
    assert trainer.net_params["cont_params"]["device"] == torch.device("cpu")
    assert trainer.net_params["cat_params"]["device"] == torch.device("cpu")


def test_saved_cuda_index_is_remapped_when_fewer_gpus_are_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert Trainer.validate_device_availability("cuda:3") == torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA GPU")
@pytest.mark.parametrize("model_name", ["tabnet", "autoint", "fttransformer"])
def test_real_gpu_training_then_cpu_prediction(model_name, monkeypatch):
    dataset = _make_cat_binary_dataset()
    algo = _make_torch_algo(model_name, torch.device("cuda:0"))
    algo.fit_predict(FoldsIterator(dataset))
    expected = algo.predict(dataset).data
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    np.testing.assert_allclose(algo.predict(dataset).data, expected, rtol=1e-4, atol=1e-5)
