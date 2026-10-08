from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from lightautoml.ml_algo.dl_model import TorchModel
from lightautoml.ml_algo.torch_based.nn_models import TabM
from lightautoml.text.embed import CatEmbedder
from lightautoml.text.embed import ContEmbedder
from lightautoml.text.nn_model import TorchUniversalModel
from lightautoml.text.nn_model import UniversalDataset


tabm = pytest.importorskip("tabm")


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini", "tabm-packed"])
@pytest.mark.parametrize("feature_type", ["cont", "cat", "mixed"])
@pytest.mark.parametrize("share_training_batches", [True, False])
@pytest.mark.parametrize("explicit_chunks", [True, False])
def test_tabm_backbone_params(arch_type, feature_type, share_training_batches, explicit_chunks, binary_task):
    torch.manual_seed(42)
    backbone_params = {"k": 4, "arch_type": arch_type, "d_block": 16, "n_blocks": 1, "dropout": 0.0}
    if explicit_chunks:
        backbone_params["start_scaling_init_chunks"] = None
    original_params = backbone_params.copy()
    cont_features = ["x0", "x1", "x2"] if feature_type in ["cont", "mixed"] else []
    cat_features = ["c0", "c1"] if feature_type in ["cat", "mixed"] else []
    frame = pd.DataFrame(
        {
            "x0": np.linspace(0, 1, 8, dtype=np.float32),
            "x1": np.linspace(1, 2, 8, dtype=np.float32),
            "x2": np.linspace(2, 3, 8, dtype=np.float32),
            "c0": np.arange(8) % 4,
            "c1": np.arange(8) % 5,
        }
    )
    data = SimpleNamespace(data=frame, target=pd.Series([0.0, 1.0] * 4), weights=None)
    algo = TorchModel(
        default_params={
            "model": "tabm",
            "device": "cpu",
            "backbone_params": backbone_params,
            "share_training_batches": share_training_batches,
            "text_features": [],
            "cont_features": cont_features,
            "cat_features": cat_features,
        }
    )
    algo.train_params = {
        "dataset": UniversalDataset,
        "bs": 4,
        "num_workers": 0,
        "pin_memory": False,
        "tokenizer": None,
        "max_length": 256,
    }
    model = TorchUniversalModel(
        task=binary_task,
        loss=binary_task.losses["torch"].loss,
        torch_model=TabM,
        cont_embedder_=ContEmbedder if cont_features else None,
        cont_params={"num_dims": 3, "input_bn": False, "embedding_size": 1},
        cat_embedder_=CatEmbedder if cat_features else None,
        cat_params={"cat_dims": [4, 5], "emb_dropout": 0.0},
        backbone_params=backbone_params,
        share_training_batches=share_training_batches,
        device="cpu",
    )

    backbone_types = {
        "tabm": tabm.MLPBackboneBatchEnsemble,
        "tabm-mini": tabm.MLPBackboneMiniEnsemble,
        "tabm-packed": tabm.MLPBackboneEnsemble,
    }
    assert isinstance(model.torch_model.backbone, backbone_types[arch_type])
    assert model.torch_model.backbone.get_original_output_shape() == (16,)
    loaders = algo.get_dataloaders_from_dicts({"train": data, "test": data})
    assert loaders["train"].batch_sampler.k == model.torch_model.backbone.k == 4
    batch = next(iter(loaders["train"]))
    assert batch["label"].shape == ((4,) if share_training_batches else (4, 4))
    loss = model(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.torch_model.output.weight.grad is not None
    assert torch.isfinite(model.torch_model.output.weight.grad).all()

    model.eval()
    with torch.no_grad():
        predictions = model.predict(next(iter(loaders["test"])))
    assert predictions.shape == (4, 1)
    assert torch.isfinite(predictions).all()
    assert backbone_params == original_params


@pytest.mark.parametrize(
    "params, expected_k",
    [({}, 32), ({"backbone_params": None}, 32), ({"k": 5}, 5), ({"backbone_params": {"k": 7}, "k": 5}, 7)],
)
def test_tabm_sampler_ensemble_size(params, expected_k):
    assert TorchModel(default_params=params)._get_tabm_k() == expected_k
