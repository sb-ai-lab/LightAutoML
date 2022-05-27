"""Neural net for tabular datasets."""

from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("nlp")


import gc
import logging
import os
import uuid

from copy import copy

import numpy as np
import torch

from torch.optim import lr_scheduler
from transformers import AutoTokenizer

from ..ml_algo.base import TabularDataset
from ..ml_algo.base import TabularMLAlgo
from ..pipelines.features.text_pipeline import _model_name_by_lang
from ..pipelines.utils import get_columns_by_role
from ..text.nn_model import CatEmbedder
from ..text.nn_model import ContEmbedder
from ..text.nn_model import TextBert
from ..text.nn_model import TorchUniversalModel
from ..text.nn_model import UniversalDataset
from ..text.trainer import Trainer
from ..text.utils import collate_dict
from ..text.utils import inv_sigmoid
from ..text.utils import inv_softmax
from ..text.utils import is_shuffle
from ..text.utils import parse_devices
from ..text.utils import seed_everything


logger = logging.getLogger(__name__)


class TorchModel(TabularMLAlgo):
    """Neural net for tabular datasets.

    default_params:

        - bs: Batch size.
        - num_workers: Number of threads for multiprocessing.
        - max_length: Max sequence length.
        - opt_params: Dict with optim params.
        - scheduler_params: Dict with scheduler params.
        - is_snap: Use snapshots.
        - snap_params: Dict with SE parameters.
        - init_bias: Init last linear bias by mean target values.
        - n_epochs: Number of training epochs.
        - input_bn: Use 1d batch norm for input data.
        - emb_dropout: Dropout probability.
        - emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        - max_emb_size: Max embedding size.
        - bert_name: Name of HuggingFace transformer model.
        - pooling: Type of pooling strategy for bert model.
        - device: Torch device or str.
        - use_cont: Use numeric data.
        - use_cat: Use category data.
        - use_text: Use text data.
        - lang: Text language.
        - deterministic: CUDNN backend.
        - multigpu: Use Data Parallel.
        - path_to_save: Path to save model checkpoints,
          ``None`` - stay in memory.
        - random_state: Random state to take subsample.
        - verbose_inside: Number of steps between
          verbose inside epoch or ``None``.
        - verbose: Verbose every N epochs.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "TorchNN"

    _default_params = {
        "bs": 16,
        "num_workers": 4,
        "max_length": 256,
        "opt_params": {
            "lr": 1e-4,
        },
        "scheduler_params": {"patience": 5, "factor": 0.5, "verbose": True},
        "is_snap": False,
        "snap_params": {"k": 1, "early_stopping": True, "patience": 1, "swa": False},
        "init_bias": True,
        "n_epochs": 20,
        "input_bn": False,
        "emb_dropout": 0.1,
        "emb_ratio": 3,
        "max_emb_size": 50,
        "bert_name": None,
        "pooling": "cls",
        "device": [0],
        "use_cont": True,
        "use_cat": True,
        "use_text": True,
        "lang": "en",
        "deterministic": True,
        "multigpu": False,
        "random_state": 42,
        "path_to_save": os.path.join("./models/", "model"),
        "verbose_inside": None,
        "verbose": 1,
    }

    def _infer_params(self):
        if self.params["path_to_save"] is not None:
            self.path_to_save = os.path.relpath(self.params["path_to_save"])
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
        else:
            self.path_to_save = None

        params = copy(self.params)
        if params["bert_name"] is None:
            params["bert_name"] = _model_name_by_lang[params["lang"]]

        params["loss"] = self.task.losses["torch"].loss
        params["metric"] = self.task.losses["torch"].metric_func

        is_text = (len(params["text_features"]) > 0) and (params["use_text"]) and (params["device"].type == "cuda")
        is_cat = (len(params["cat_features"]) > 0) and (params["use_cat"])
        is_cont = (len(params["cont_features"]) > 0) and (params["use_cont"])

        model = Trainer(
            net=TorchUniversalModel,
            net_params={
                "loss": params["loss"],
                "task": self.task,
                "n_out": params["n_out"],
                "cont_embedder": ContEmbedder if is_cont else None,
                "cont_params": {
                    "num_dims": params["cont_dim"],
                    "input_bn": params["input_bn"],
                }
                if is_cont
                else None,
                "cat_embedder": CatEmbedder if is_cat else None,
                "cat_params": {
                    "cat_dims": params["cat_dims"],
                    "emb_dropout": params["emb_dropout"],
                    "emb_ratio": params["emb_ratio"],
                    "max_emb_size": params["max_emb_size"],
                }
                if is_cat
                else None,
                "text_embedder": TextBert if is_text else None,
                "text_params": {
                    "model_name": params["bert_name"],
                    "pooling": params["pooling"],
                }
                if is_text
                else None,
                "bias": params["bias"],
            },
            opt=torch.optim.Adam,
            opt_params=params["opt_params"],
            n_epochs=params["n_epochs"],
            device=params["device"],
            device_ids=params["device_ids"],
            is_snap=params["is_snap"],
            snap_params=params["snap_params"],
            sch=lr_scheduler.ReduceLROnPlateau,
            scheduler_params=params["scheduler_params"],
            verbose=params["verbose"],
            verbose_inside=params["verbose_inside"],
            metric=params["metric"],
            apex=False,
        )

        self.train_params = {
            "dataset": UniversalDataset,
            "bs": params["bs"],
            "num_workers": params["num_workers"],
            "tokenizer": AutoTokenizer.from_pretrained(params["bert_name"], use_fast=False) if is_text else None,
            "max_length": params["max_length"],
        }

        return model

    @staticmethod
    def get_mean_target(target, task_name):
        bias = (
            np.array(target.mean(axis=0)).reshape(1, -1).astype(float)
            if (task_name != "multiclass")
            else np.unique(target, return_counts=True)[1]
        )
        bias = (
            inv_sigmoid(bias)
            if (task_name == "binary") or (task_name == "multilabel")
            else inv_softmax(bias)
            if (task_name == "multiclass")
            else bias
        )

        bias[bias == np.inf] = np.nanmax(bias[bias != np.inf])
        bias[bias == -np.inf] = np.nanmin(bias[bias != -np.inf])
        bias[bias == np.NaN] = np.nanmean(bias[bias != np.NaN])

        return bias

    def init_params_on_input(self, train_valid_iterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        suggested_params = copy(self.default_params)
        suggested_params["device"], suggested_params["device_ids"] = parse_devices(
            suggested_params["device"], suggested_params["multigpu"]
        )

        task_name = train_valid_iterator.train.task.name
        target = train_valid_iterator.train.target
        suggested_params["n_out"] = 1 if task_name != "multiclass" else np.max(target) + 1

        cat_dims = []
        suggested_params["cat_features"] = get_columns_by_role(train_valid_iterator.train, "Category")
        for cat_feature in suggested_params["cat_features"]:
            num_unique_categories = max(train_valid_iterator.train[:, cat_feature].data) + 1
            cat_dims.append(num_unique_categories)
        suggested_params["cat_dims"] = cat_dims

        suggested_params["cont_features"] = get_columns_by_role(train_valid_iterator.train, "Numeric")
        suggested_params["cont_dim"] = len(suggested_params["cont_features"])

        suggested_params["text_features"] = get_columns_by_role(train_valid_iterator.train, "Text")
        suggested_params["bias"] = self.get_mean_target(target, task_name) if suggested_params["init_bias"] else None

        return suggested_params

    def get_dataloaders_from_dicts(self, data_dict):
        logger.debug(f'number of text features: {len(self.params["text_features"])} ')
        logger.debug(f'number of categorical features: {len(self.params["cat_features"])} ')
        logger.debug(f'number of continuous features: {self.params["cont_dim"]} ')

        datasets = {}
        for stage, value in data_dict.items():
            data = {
                name: value.data[cols].values
                for name, cols in zip(
                    ["text", "cat", "cont"],
                    [
                        self.params["text_features"],
                        self.params["cat_features"],
                        self.params["cont_features"],
                    ],
                )
                if len(cols) > 0
            }

            datasets[stage] = self.train_params["dataset"](
                data=data,
                y=value.target.values if stage != "test" else np.ones(len(value.data)),
                w=value.weights.values if value.weights is not None else np.ones(len(value.data)),
                tokenizer=self.train_params["tokenizer"],
                max_length=self.train_params["max_length"],
                stage=stage,
            )

        dataloaders = {
            stage: torch.utils.data.DataLoader(
                datasets[stage],
                batch_size=self.train_params["bs"],
                shuffle=is_shuffle(stage),
                num_workers=self.train_params["num_workers"],
                collate_fn=collate_dict,
                pin_memory=False,
            )
            for stage, value in data_dict.items()
        }
        return dataloaders

    def fit_predict_single_fold(self, train, valid):
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        seed_everything(self.params["random_state"], self.params["deterministic"])
        model = self._infer_params()

        model_path = (
            os.path.join(self.path_to_save, f"{uuid.uuid4()}.pickle") if self.path_to_save is not None else None
        )
        # init datasets
        dataloaders = self.get_dataloaders_from_dicts({"train": train, "val": valid})

        val_pred = model.fit(dataloaders)

        if model_path is None:
            model_path = model.state_dict(model_path)
        else:
            model.state_dict(model_path)

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()
        return model_path, val_pred

    def predict_single_fold(self, model: any, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Neural net object or dict or str.
            dataset: Test dataset.

        Return:
            Predicted target values.

        """

        seed_everything(self.params["random_state"], self.params["deterministic"])
        dataloaders = self.get_dataloaders_from_dicts({"test": dataset})

        if isinstance(model, (str, dict)):
            model = self._infer_params().load_state(model)

        pred = model.predict(dataloaders["test"], "test")

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()

        return pred
