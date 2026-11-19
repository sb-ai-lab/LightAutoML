"""Real CatBoost multioutput GPU training followed by CPU-only inference."""

import os
import pickle
import subprocess
import sys

from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
import pytest
import torch

from catboost.utils import get_gpu_device_count
from packaging.version import Version
from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import FoldsIterator


@pytest.fixture()
def catboost_gpu():
    try:
        available = get_gpu_device_count() > 0
    except cb.CatBoostError:
        available = False
    if not available:
        pytest.skip("Requires a real CUDA GPU for CatBoost training")
    if Version(cb.__version__) < Version("1.2.1"):
        pytest.skip("GPU tests for both tasks require CatBoost >= 1.2.1")


def _task(name):
    if name == "multilabel":
        return Task(name, metric=roc_auc_score, greater_is_better=True)
    return Task(name)


def _data(name, cardinality):
    rng = np.random.RandomState(42)
    data = rng.normal(size=(128, 3)).astype(np.float32)
    target = np.column_stack([data[:, 0] + data[:, 1], data[:, 2] - data[:, 1]])
    if name == "multilabel":
        target = (target > 0).astype(np.float32)
    roles = {name: NumericRole() for name in ["a", "b", "c"]}
    if cardinality:
        data = np.column_stack([data, np.arange(128) % cardinality])
        roles["category"] = CategoryRole(np.int32, label_encoded=True)
    return NumpyDataset(data, list(roles), roles, task=_task(name), target=target, folds=np.arange(128) % 2)


def _check_cpu_artifact(path):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert get_gpu_device_count() == 0
    with open(path, "rb") as stream:
        model, data, expected = pickle.load(stream)
    actual = model.predict(data).data
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def _cpu_roundtrip(model, data, tmp_path):
    expected = model.predict(data).data
    artifact = tmp_path / "catboost_gpu.pkl"
    with artifact.open("wb") as stream:
        pickle.dump((model, data, expected), stream)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.integration.test_catboost_multitask_gpu import _check_cpu_artifact; "
            "import sys; _check_cpu_artifact(sys.argv[1])",
            str(artifact),
        ],
        cwd=root,
        env=env,
        check=True,
        timeout=90,
    )


@pytest.mark.parametrize("task_name", ["multi:reg", "multilabel"])
@pytest.mark.parametrize("cardinality,one_hot_max_size", [(0, 2), (4, 32), (24, 2)])
def test_catboost_gpu_training_cpu_prediction(task_name, cardinality, one_hot_max_size, catboost_gpu, tmp_path):
    dataset = _data(task_name, cardinality)
    model = BoostCB(
        default_params={
            "task_type": "GPU",
            "devices": "0",
            "gpu_ram_part": 0.1,
            "num_trees": 4,
            "max_depth": 3,
            "od_wait": 2,
            "thread_count": 2,
            "one_hot_max_size": one_hot_max_size,
        },
        freeze_defaults=True,
    )
    oof = model.fit_predict(FoldsIterator(dataset))
    assert oof.shape == (128, 2)
    assert np.isfinite(oof.data).all()
    assert all(fold.get_all_params()["task_type"] == "GPU" for fold in model.models)
    assert all(bool(fold.get_cat_feature_indices()) == bool(cardinality) for fold in model.models)
    _cpu_roundtrip(model, dataset, tmp_path)


@pytest.mark.parametrize("task_name", ["multi:reg", "multilabel"])
@pytest.mark.parametrize("cardinality", [0, 24])
def test_tabular_preset_gpu_selection(task_name, cardinality, catboost_gpu, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Preset GPU selection requires CUDA-enabled PyTorch")
    dataset = _data(task_name, cardinality)
    frame = pd.DataFrame(dataset.data, columns=dataset.features)
    if cardinality:
        frame["category"] = frame["category"].astype(str)
    frame[["y0", "y1"]] = dataset.target
    model = TabularAutoML(
        task=dataset.task,
        gpu_ids="0",
        cpu_limit=2,
        timeout=60,
        debug=True,
        general_params={"use_algos": [["cb"]]},
        reader_params={"cv": 2, "advanced_roles": False},
        selection_params={"mode": 0},
        gbm_pipeline_params={"output_categories": True},
        cb_params={
            "default_params": {"num_trees": 4, "max_depth": 3, "od_wait": 2, "one_hot_max_size": 2, "gpu_ram_part": 0.1}
        },
    )
    roles = {"target": ["y0", "y1"]}
    if cardinality:
        roles["category"] = ["category"]
    oof = model.fit_predict(frame, roles=roles, verbose=0)
    assert oof.shape == (128, 2)
    assert np.isfinite(oof.data).all()
    models = [fold for level in model.levels for pipe in level for algo in pipe.ml_algos for fold in algo.models]
    assert models and all(fold.get_all_params()["task_type"] == "GPU" for fold in models)
    assert all(bool(fold.get_cat_feature_indices()) == bool(cardinality) for fold in models)
    _cpu_roundtrip(model, frame.drop(columns=["y0", "y1"]), tmp_path)
