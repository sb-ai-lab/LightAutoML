"""Neural net for tabular datasets."""

from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("nlp")


import gc
import logging
import os
import uuid

from copy import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from transformers import AutoTokenizer
from .nn_models import DenseLightModel, DenseModel, ResNetModel, MLP, LinearLayer, SNN
from ..utils.timer import TaskTimer
from typing import Optional

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

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from lightautoml.tasks.losses.torch import TorchLossWrapper

from lightautoml.validation.base import TrainValidIterator
from lightautoml.dataset.np_pd_dataset import NumpyDataset

logger = logging.getLogger(__name__)

model_by_name = {'dense_light': DenseLightModel, 'dense': DenseModel, 'resnet': ResNetModel,
                 'mlp': MLP, 'dense_layer': LinearLayer, 'snn': SNN}


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
    _name: str = 'TorchNN'
    _params: Dict = None
    
    _default_models_params = {
        'use_noise': False,
        'use_bn': True,
        'use_dropout': True,
        
        'act_fun': nn.ReLU,
        'drop_rate_base': 0.1,
        
        'num_layers': 3,
        'hidden_size_base': 512,
        'num_blocks': 2,
        'block_size_base': 2,
        'growth_size': 256,
        'bn_factor': 2,
        'compression': 0.5,
        'efficient': False,
        'hid_factor_base': 2,
        'drop_rate_base_1': 0.1,
        'drop_rate_base_2': 0.2,
        'hidden_size': 512,
        'drop_rate': 0.1
    }
    
    _default_params = {
        'num_workers': 4,
        'max_length': 256,
        'is_snap': False,
        'input_bn': False,
        'max_emb_size': 256,
        'bert_name': None,
        'pooling': 'cls',
        'device': torch.device('cuda:0'),
        'use_cont': True,
        'use_cat': True,
        'use_text': True,
        'lang': 'en',
        'deterministic': True,
        'multigpu': False,
        'random_state': 42,
        'model': 'dense',
        'path_to_save': os.path.join('./models/', 'model'),
        'verbose_inside': None,
        'verbose': 1,

        'n_epochs': 20,
        'snap_params': {'k': 3, 'early_stopping': True, 'patience': 16, 'swa': True},
        'bs': 512,
        'drop_last_batch': True,
        'emb_dropout': 0.1,
        'emb_ratio': 3,
        'opt': torch.optim.Adam,
        'opt_params': {
            'weight_decay': 0,
            'lr': 3e-4
        },
        'sch': ReduceLROnPlateau,
        'scheduler_params': {
            'patience': 10,
            'factor': 1e-2,
            'min_lr': 1e-5
        },
        'init_bias': True,
        **_default_models_params
    }

    _task_to_loss = {
        'binary': nn.BCEWithLogitsLoss(),
        'multiclass': TorchLossWrapper(nn.CrossEntropyLoss, True, False),
        'reg': nn.MSELoss()
    }

    def __init__(
        self,
        default_params: Optional[dict] = None,
        timer: Optional[TaskTimer] = None,
        optimization_search_space: Optional[dict] = {},
    ):
        super().__init__(default_params, False, timer, optimization_search_space)
        if not self.optimization_search_space:
            self.optimization_search_space = TorchModel._default_sample

    def _default_sample(optimization_search_space, trial, suggested_params):
        trial_values = copy(suggested_params)

        trial_values["lr"] = trial.suggest_loguniform(
            "lr", low=1e-5, high=1e-1
        )
        trial_values["bs"] = trial.suggest_categorical(
            "bs", [2 ** i for i in range(6, 11)]
        )
        weight_decay_bin = trial.suggest_categorical(
            "weight_decay_bin", [0, 1]
        )

        if weight_decay_bin == 0:
            trial_values["weight_decay"] = 0
        else:
            trial_values["weight_decay"] = trial.suggest_loguniform(
                "weight_decay", low=1e-6, high=1e-2
            )

        trial_values["opt_params"] = {
            "lr": trial_values["lr"],
            "weight_decay": trial_values["weight_decay"]
        }
        return trial_values


    @property
    def params(self) -> dict:
        """Get model's params dict."""
        if self._params is None:
            self._params = copy(self.default_params)
        return self._params

    @params.setter
    def params(self, new_params: dict):
        assert isinstance(new_params, dict)
        self._params = {**self.params, **new_params}
        self._params = {**self._params, **self._construct_tune_params(self._params)}

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

        if self.params.get("loss", False):
            self.custom_loss = True
            params["loss"] = self.params["loss"]
        else:
            self.custom_loss = False
            if self.task._name in self._task_to_loss:
                params["loss"] = self._task_to_loss[self.task._name]
                self.custom_loss = True
            else:
                params["loss"] = self.task.losses["torch"]

        params["custom_loss"] = self.custom_loss
        params["metric"] = self.task.losses["torch"].metric_func

        is_text = (len(params["text_features"]) > 0) and (params["use_text"]) and (params["device"].type == "cuda")
        is_cat = (len(params["cat_features"]) > 0) and (params["use_cat"])
        is_cont = (len(params["cont_features"]) > 0) and (params["use_cont"])

        torch_model = params["model"]

        if isinstance(torch_model, str):
            assert torch_model in model_by_name, "Wrong model name"
            torch_model = model_by_name[torch_model]
        else:
            for k in self._default_models_params.keys():
                if k in params:
                    del params[k]

        assert issubclass(torch_model, nn.Module), "Wrong model format, only support torch models"

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
                    "device": params["device"]
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
                "torch_model": torch_model,
                **params
            },
            apex=False,
            **params
        )

        self.train_params = {
            "dataset": UniversalDataset,
            "bs": params["bs"],
            "drop_last": params["drop_last_batch"],
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

        # Cat_features are needed to be preprocessed with LE, where 0 = not known category
        valid = train_valid_iterator.get_validation_data()
        for cat_feature in suggested_params["cat_features"]:
            num_unique_categories = max(max(train_valid_iterator.train[:, cat_feature].data), max(valid[:, cat_feature].data)) + 1
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
        
        print(self.train_params)

        dataloaders = {
            stage: torch.utils.data.DataLoader(
                datasets[stage],
                batch_size=self.train_params["bs"],
                shuffle=is_shuffle(stage),
                num_workers=self.train_params["num_workers"],
                drop_last=self.train_params["drop_last"] if stage == "train" else False,
                collate_fn=collate_dict,
                pin_memory=False,
            )
            for stage, value in data_dict.items()
        }
        return dataloaders

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> NumpyDataset:
        if "cont_features" not in self.params:
            self.params = self.init_params_on_input(train_valid_iterator)

        return super().fit_predict(train_valid_iterator)

    def fit_predict_single_fold(self, train, valid):
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        seed_everything(self.params["random_state"], self.params["deterministic"])
        task_name = train.task.name
        target = train.target
        self.params['bias'] = self.get_mean_target(target, task_name) if self.params['init_bias'] else None

        model = self._infer_params()

        model_path = (
            os.path.join(self.path_to_save, f"{uuid.uuid4()}.pickle") if self.path_to_save is not None else None
        )
        # init datasets
        dataloaders = self.get_dataloaders_from_dicts({"train": train.to_pandas(), "val": valid.to_pandas()})

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
        dataloaders = self.get_dataloaders_from_dicts({"test": dataset.to_pandas()})

        if isinstance(model, (str, dict)):
            model = self._infer_params().load_state(model)

        pred = model.predict(dataloaders["test"], "test")

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()

        return pred

    def _construct_tune_params(self, params):
        new_params = {}

        new_params["opt_params"] = params.get("opt_params", dict())
        if "lr" in params:
            new_params["opt_params"]["lr"] = params["lr"]
        if "weight_decay" in params:
            new_params["opt_params"]["weight_decay"] = params["weight_decay"]
        elif params.get("weight_decay_bin", -1) == 0:
            new_params["opt_params"]["weight_decay"] = 0

        new_params["scheduler_params"] = params.get("scheduler_params", dict())
        if params["sch"] == StepLR:
            if "step_size" in params:
                new_params["scheduler_params"]["step_size"] = params["step_size"]
            if "gamma" in params:
                new_params["scheduler_params"]["gamma"] = params["gamma"]

            new_params["scheduler_params"] = {
                "step_size": new_params["scheduler_params"]["step_size"],
                "gamma": new_params["scheduler_params"]["gamma"],
            }

        elif params["sch"] == ReduceLROnPlateau:
            if "patience" in params:
                new_params["scheduler_params"]["patience"] = params["patience"]
            if "factor" in params:
                new_params["scheduler_params"]["factor"] = params["factor"]

            new_params["scheduler_params"] = {
                "patience": new_params["scheduler_params"]["patience"],
                "factor": new_params["scheduler_params"]["factor"],
                'min_lr': 1e-6,
            }

        elif params["sch"] == CosineAnnealingLR:
            if "T_max" in params:
                new_params["scheduler_params"]["T_max"] = params["T_max"]
            if "eta_min" in params:
                new_params["scheduler_params"]["eta_min"] = params["eta_min"]
            elif params.get("eta_min_bin", -1) == 0:
                new_params["scheduler_params"]["eta_min"] = 0

            new_params["scheduler_params"] = {
                "T_max": new_params["scheduler_params"]["T_max"],
                "eta_min": new_params["scheduler_params"]["eta_min"],
            }

        elif params["sch"] is not None:
            raise ValueError("Worng sch")

        if self.params["model"] == "dense_light" or self.params["model"] == "mlp":
            hidden_size = ()
            drop_rate = ()

            for layer in range(int(params["num_layers"])):
                hidden_name = "hidden_size_base"
                drop_name = "drop_rate_base"

                hidden_size = hidden_size + (params[hidden_name],)
                if self.params["use_dropout"]:
                    drop_rate = drop_rate + (params[drop_name],)

            new_params["hidden_size"] = hidden_size
            if self.params["use_dropout"]:
                new_params["drop_rate"] = drop_rate

        elif self.params["model"] == "dense":
            block_config = ()
            drop_rate = ()

            for layer in range(int(params["num_blocks"])):
                block_name = "block_size_base"
                drop_name = "drop_rate_base"

                block_config = block_config + (params[block_name],)
                if self.params["use_dropout"]:
                    drop_rate = drop_rate + (params[drop_name],)

            new_params["block_config"] = block_config
            if self.params["use_dropout"]:
                new_params["drop_rate"] = drop_rate

        elif self.params["model"] == "resnet":
            hidden_factor = ()
            drop_rate = ()

            for layer in range(int(params['num_layers'])):
                hidden_name = "hid_factor_base"
                drop_name = "drop_rate_base"

                hidden_factor = hidden_factor + (params[hidden_name],)
                if self.params["use_dropout"]:
                    drop_rate = drop_rate + ((params[drop_name + '_1'], params[drop_name + '_2']),)

            new_params['hid_factor'] = hidden_factor
            if self.params["use_dropout"]:
                new_params["drop_rate"] = drop_rate

        return new_params
