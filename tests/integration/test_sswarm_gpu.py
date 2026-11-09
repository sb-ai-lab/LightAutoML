"""GPU training followed by SSWARM with GPU or CPU model inference."""

import json
import os
import pickle
import subprocess
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lightautoml.addons.tabular_interpretation import SSWARM
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_xgb import BoostXGB
from lightautoml.ml_algo.dl_model import TorchModel
from lightautoml.tasks import Task


def _explain(model, data):
    explainer = SSWARM(model, random_state=77)
    values = explainer.shap_values(data, T=40, n_repeats=1, batch_size=128, n_jobs=1)
    prediction = model.predict(data).data
    probabilities = np.column_stack([1 - prediction[:, 0], prediction[:, 0]])
    assert values.shape == (2, len(data), len(model.reader.used_features))
    assert np.isfinite(values).all()
    np.testing.assert_allclose(
        values.sum(axis=-1) + explainer.expected_value[:, None], probabilities.T, rtol=1e-5, atol=1e-6
    )
    return values, prediction


def _check_cpu_inference(path):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_available()
    with open(path, "rb") as stream:
        model, data, gpu_values, gpu_prediction = pickle.load(stream)
    values, prediction = _explain(model, data)
    np.testing.assert_allclose(values, gpu_values, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(prediction, gpu_prediction, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA GPU for training")
@pytest.mark.parametrize("algorithm", ["cb", "xgb", "mlp", "fttransformer"])
def test_gpu_training_then_sswarm_on_cpu(algorithm, tmp_path):
    data = pd.DataFrame(
        np.random.default_rng(123).normal(size=(320, 6)).astype(np.float32), columns=[f"f{i}" for i in range(6)]
    )
    data["target"] = (data["f0"] + 0.7 * data["f1"] - data["f2"] > 0).astype(int)
    model = TabularAutoML(
        task=Task("binary"),
        timeout=60,
        cpu_limit=2,
        gpu_ids="0",
        debug=True,
        general_params={"use_algos": [[algorithm]]},
        reader_params={"cv": 2, "advanced_roles": False, "random_state": 123},
        cb_params={"default_params": {"num_trees": 12, "max_depth": 4, "od_wait": 3, "random_seed": 123}},
        xgb_params={"default_params": {"n_estimators": 12, "max_depth": 3, "early_stopping_rounds": 3, "seed": 123}},
        nn_params={
            "n_epochs": 2,
            "bs": 64,
            "num_workers": 0,
            "path_to_save": None,
            "freeze_defaults": True,
            "hidden_size": [32, 16],
            "random_state": 123,
            "multigpu": False,
        },
    )
    observed_devices = set()

    def record_device(module, inputs, output):
        if isinstance(output, torch.Tensor):
            observed_devices.add(output.device.type)

    handle = torch.nn.modules.module.register_module_forward_hook(record_device)
    try:
        model.fit_predict(data, roles={"target": "target"}, verbose=0)
        if algorithm in ("mlp", "fttransformer"):
            assert "cuda" in observed_devices
        observed_devices.clear()
        observations = data.iloc[:16].drop(columns="target")
        gpu_values, gpu_prediction = _explain(model, observations)
        if algorithm in ("mlp", "fttransformer"):
            assert observed_devices == {"cuda"}
    finally:
        handle.remove()

    for level in model.levels:
        for pipe in level:
            for algo in pipe.ml_algos:
                if isinstance(algo, BoostCB):
                    # CatBoost's prediction evaluator defaults to CPU even after GPU training.
                    assert all(fold.get_all_params()["task_type"] == "GPU" for fold in algo.models)
                elif isinstance(algo, BoostXGB):
                    for booster in algo.models:
                        assert json.loads(booster.save_config())["learner"]["generic_param"]["device"].startswith(
                            "cuda"
                        )
                        booster.set_param({"device": "cpu"})
                    algo.params["device"] = "cpu"
                elif isinstance(algo, TorchModel):
                    assert algo.params["device"].type == "cuda"
                    algo.params.update(device=torch.device("cpu"), device_ids=None, multigpu=False)

    cpu_values, cpu_prediction = _explain(model, observations)
    np.testing.assert_allclose(cpu_values, gpu_values, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cpu_prediction, gpu_prediction, rtol=1e-4, atol=1e-5)

    # Save after selecting CPU inference; trained weights remain unchanged.
    # This also avoids GPU auto-fallback warnings when unpickling XGBoost.
    artifact = tmp_path / "gpu_trained_cpu_inference.pkl"
    with artifact.open("wb") as stream:
        pickle.dump((model, observations, gpu_values, gpu_prediction), stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-c",
            "from tests.integration.test_sswarm_gpu import _check_cpu_inference; "
            "import sys; _check_cpu_inference(sys.argv[1])",
            str(artifact),
        ],
        cwd=root,
        env=env,
        check=True,
        timeout=120,
    )
