"""Main pytorch training and prediction class with Snapshots Ensemble."""

import logging

from copy import deepcopy
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from .dp_utils import CustomDataParallel


try:
    from apex import amp
except:
    amp = None


from .utils import _dtypes_mapping


logger = logging.getLogger(__name__)


def optim_to_device(optim: torch.optim.Optimizer, device: torch.device) -> torch.optim.Optimizer:
    """Change optimizer device.

    Args:
        optim: Optimizer.
        device: To device.

    Returns:
        Optimizer on selected device.

    """

    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return optim


class SnapshotEns:
    """In memory snapshots class."""

    def __init__(
        self,
        device: torch.device,
        k: int = 1,
        early_stopping: bool = True,
        patience: int = 3,
        swa: bool = False,
    ):
        """Class for SE, SWA and early stopping.

        Args:
            device: Torch device.
            k: Number of snapshots / checkpoint for swa.
            early_stopping: Use early stopping.
            patience: Patience before early stopping.
            swa: Use stochastic weight averaging.

        """
        self.best_loss = np.array([np.inf] * k)
        self.k = k
        self.device = device
        self.models = [nn.Module()] * k
        self.early_stopping = early_stopping
        self.patience = patience
        self.swa = swa
        self.counter = 0
        self.early_stop = False

    def update(self, model: nn.Module, loss: float):
        """Update current state.

        Args:
            model: Torch model.
            loss: Loss value, lower is better.

        """
        if np.any(self.best_loss > loss):
            self._sort()
            pos = np.where(self.best_loss > loss)[0][-1]

            self.best_loss[pos] = loss
            self.models[pos] = deepcopy(model.eval()).cpu()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True and self.early_stopping

    def predict(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make snapshots prediction.

        Final prediction is a result of averaging predictions from snapshots.

        Args:
            data: Dict with model data.

        Returns:
            Torch tensor snapshot ensemble prediction.

        """

        preds = 0
        for model in self.models:
            model.eval().to(self.device)
            preds += model.predict(data)
            model.cpu()

        preds /= self.k
        return preds

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make snapshots forward (compute loss).

        Final result is a result of averaging losses from snapshots.

        Args:
            data: Dict with model data.

        Returns:
            Torch tensor snapshot ensemble loss.

        """
        preds = 0
        for model in self.models:
            model.eval().to(self.device)
            preds += model.forward(data)
            model.cpu()
        preds /= self.k
        return preds

    def _sort(self):
        """Sort models by loss value."""

        ids = np.argsort(self.best_loss)
        self.best_loss = self.best_loss[ids]
        self.models = [self.models[z] for z in ids]

    def set_weights(self, model: nn.Module, best: bool = False):
        """Set model weights as SWA or from best state.

        Args:
            model: Torch model.
            best: Save only best model.

        """
        n = 1 if best else min(self.k, sum(self.best_loss != np.inf))
        state_dict = {}
        for pos, score in enumerate(self.best_loss):

            if pos == n:
                break

            w = 1 / n

            new_state = self.models[pos].state_dict()
            # upd new state with weights
            for i in new_state.keys():
                new_state[i] = new_state[i].double() * w

            if pos == 0:
                state_dict = new_state
            else:
                # upd state
                for i in state_dict.keys():
                    state_dict[i] += new_state[i]

        model.load_state_dict(state_dict)

    def set_best_params(self, model: nn.Module):
        """Set best model params and clean cache.

        Args:
            model: Torch model.

        """
        self._sort()
        self.set_weights(model, best=False if self.swa else True)

        # TODO: think about dropping all models if use SWA. Change state_dict and load_state_dict
        # drop empty slots
        min_k = min(self.k, sum(self.best_loss != np.inf))
        self.models = self.models[:min_k]
        self.best_loss = self.best_loss[:min_k]

    def state_dict(self) -> Dict:
        """State SE dict.

        Returns:
            Dict with SE state.

        """
        models_dict = {"best_loss": self.best_loss}

        for n, model in enumerate(self.models):
            if isinstance(model, CustomDataParallel):
                models_dict[n] = model.module.state_dict()
            else:
                models_dict[n] = model.state_dict()
        return models_dict

    def load_state_dict(self, weights: Dict, model: nn.Module):
        """Load SE state.

        Args:
            model: Torch model.
            weights: State dict with weights.

        """
        self.best_loss = weights.pop("best_loss")
        self.models = [nn.Module()] * len(self.best_loss)
        for key, model_weights in weights.items():
            if isinstance(model, CustomDataParallel):
                model.module.load_state_dict(model_weights)
            else:
                model.load_state_dict(model_weights)
            self.models[key] = deepcopy(model.eval()).cpu()

        return self


class Trainer:
    """Torch main trainer class."""

    def __init__(
        self,
        net,
        net_params: Dict,
        opt,
        opt_params: Dict,
        n_epochs: int,
        device: torch.device,
        device_ids: List[int],
        metric: Callable,
        snap_params: Dict,
        is_snap: bool = False,
        sch: Optional = None,
        scheduler_params: Optional[Dict] = None,
        verbose: int = 1,
        verbose_inside: Optional[int] = None,
        apex: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        """Train, validation and test loops for NN models.

        Use DataParallel if device_ids is not None.

        Args:
            net: Uninitialized torch model.
            net_params: Dict with model params.
            opt: Uninitialized torch optimizer.
            opt_params: Dict with optim params.
            n_epochs: Number of training epochs.
            device: Torch device.
            device_ids: Ids of used gpu devices or None.
            metric: Callable metric for validation.
            snap_params: Dict with SE parameters.
            is_snap: Use snapshots.
            sch: Uninitialized torch scheduler.
            scheduler_params: Dict with scheduler params.
            verbose: Verbose every N epochs.
            verbose_inside: Number of steps between verbose
              inside epoch or None.
            apex: Use apex (lead to GPU memory leak among folds).
            pretrained_path: Path to the pretrained model weights.

        """
        self.net = net
        self.net_params = net_params
        self.opt = opt
        self.opt_params = opt_params
        self.n_epochs = n_epochs
        self.device = device
        self.device_ids = device_ids
        self.is_snap = is_snap
        self.snap_params = snap_params
        self.sch = sch
        self.scheduler_params = scheduler_params
        self.verbose = verbose
        self.metric = metric
        self.verbose_inside = verbose_inside
        self.apex = apex
        self.pretrained_path = pretrained_path

        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.se = None
        self.amp = None
        self.is_fitted = False

    def clean(self):
        """Clean all models."""

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.se = None
        self.amp = None
        return self

    def _init(self):
        """Init all models."""

        self.model = self.net(**self.net_params)
        if self.device_ids is not None:
            self.model = CustomDataParallel(self.model, device_ids=self.device_ids)

        self.se = SnapshotEns(self.device, **self.snap_params)
        self.optimizer = self.opt(self.model.parameters(), **self.opt_params)
        self.amp = amp if self.apex else None
        if self.amp is not None:
            opt_level = "O1"
            self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level=opt_level)
        self.model.to(self.device)
        self.scheduler = self.sch(self.optimizer, **self.scheduler_params) if self.sch is not None else None
        return self

    def load_state(self, path: Union[str, Dict]):
        """Load all models state.

        Args:
            path: Path to state dict or state dict.

        """

        if isinstance(path, str):
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
            self.pretrained_path = path
        else:
            checkpoint = path

        self._init()
        if checkpoint["se"] is not None:
            self.se.load_state_dict(checkpoint["se"], self.model)

        if isinstance(self.model, CustomDataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimizer = optim_to_device(self.optimizer, self.device)

        if checkpoint["amp"] is not None:
            self.amp.load_state_dict(checkpoint["amp"])

        self.is_fitted = True

        del checkpoint, path
        return self

    def state_dict(self, path: Optional = None):
        """Create all models state.

        Switch all models on cpu before saving state.

        Args:
            path: Path to save state dict.

        Returns:
            Checkpoint if path is not None.

        """

        self.model.to(torch.device("cpu"))
        self.optimizer = optim_to_device(self.optimizer, torch.device("cpu"))
        if isinstance(self.model, CustomDataParallel):
            model_checkpoint = self.model.module.state_dict()
        else:
            model_checkpoint = self.model.state_dict()

        checkpoint = {
            "model": model_checkpoint,
            "optimizer": self.optimizer.state_dict(),
            "amp": self.amp.state_dict() if self.apex else None,
            "se": self.se.state_dict() if self.is_snap else None,
        }
        if path is not None:
            torch.save(checkpoint, path)
            del checkpoint
            return self
        else:
            return checkpoint

    def fit(self, dataloaders: Dict[str, DataLoader]) -> np.ndarray:
        """Fit model.

        Args:
            dataloaders: Dict with torch dataloaders.

        Returns:
            Validation prediction.

        """

        if self.pretrained_path is not None:
            self.load_state(self.pretrained_path)
        elif (self.model is None) or (not self.is_fitted):
            self._init()

        train_log = []
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            # train
            train_loss = self.train(dataloaders=dataloaders)
            train_log.extend(train_loss)
            # test
            val_loss, val_data = self.test(dataloader=dataloaders["val"])
            self.se.update(self.model, np.mean(val_loss))

            if self.sch is not None:
                self.scheduler.step(np.mean(val_loss))

            if (self.verbose is not None) and ((epoch + 1) % self.verbose == 0):
                logger.info3(
                    "Epoch: {e}, train loss: {tl}, val loss: {vl}, val metric: {me}".format(
                        me=self.metric(*val_data),
                        e=self.epoch,
                        tl=np.mean(train_loss),
                        vl=np.mean(val_loss),
                    )
                )
            if self.se.early_stop:
                break

        self.se.set_best_params(self.model)

        if self.is_snap:
            val_loss, val_data = self.test(dataloader=dataloaders["val"], snap=True, stage="val")
            logger.info3(
                "Result SE, val loss: {vl}, val metric: {me}".format(me=self.metric(*val_data), vl=np.mean(val_loss))
            )
        elif self.se.early_stop:
            val_loss, val_data = self.test(dataloader=dataloaders["val"])
            logger.info3(
                "Early stopping: val loss: {vl}, val metric: {me}".format(
                    me=self.metric(*val_data), vl=np.mean(val_loss)
                )
            )

        self.is_fitted = True

        return val_data[1]

    def train(self, dataloaders: Dict[str, DataLoader]) -> List[float]:
        """Training loop.

        Args:
            dataloaders: Dict with torch dataloaders.

        Returns:
            Loss.

        """

        loss_log = []
        self.model.train()
        running_loss = 0
        c = 0
        logging_level = logger.getEffectiveLevel()
        if logging_level <= logging.INFO and self.verbose:
            loader = tqdm(dataloaders["train"], desc="train", disable=False)
        else:
            loader = dataloaders["train"]

        for sample in loader:
            data = {
                i: (sample[i].long().to(self.device) if _dtypes_mapping[i] == "long" else sample[i].to(self.device))
                for i in sample.keys()
            }

            loss = self.model(data).mean()
            if self.apex:
                with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = loss.data.cpu().numpy()
            loss_log.append(loss)
            running_loss += loss

            c += 1
            if self.verbose_inside and logging_level <= logging.INFO:
                if c % self.verbose_inside == 0:
                    val_loss, val_data = self.test(dataloader=dataloaders["val"])
                    self.se.update(self.model, np.mean(val_loss))
                    if self.verbose is not None:
                        logger.info3(
                            "Epoch: {e}, iter: {c}, val loss: {vl}, val metric: {me}".format(
                                me=self.metric(*val_data),
                                e=self.epoch,
                                c=c,
                                vl=np.mean(val_loss),
                            )
                        )
            if logging_level <= logging.INFO and self.verbose:
                loader.set_description("train (loss=%g)" % (running_loss / c))

        return loss_log

    def test(
        self, dataloader: DataLoader, stage: str = "val", snap: bool = False
    ) -> Tuple[List[float], Tuple[np.ndarray, np.ndarray]]:
        """Testing loop.

        Args:
            dataloader: Torch dataloader.
            stage: Train, val or test.
            snap: Use snapshots.

        Returns:
            Loss, (Target, OOF).

        """
        loss_log = []
        self.model.eval()
        pred = []
        target = []
        logging_level = logger.getEffectiveLevel()
        if logging_level <= logging.INFO and self.verbose:
            loader = tqdm(dataloader, desc=stage, disable=False)
        else:
            loader = dataloader

        with torch.no_grad():
            for sample in loader:
                data = {
                    i: (sample[i].long().to(self.device) if _dtypes_mapping[i] == "long" else sample[i].to(self.device))
                    for i in sample.keys()
                }

                if snap:
                    output = self.se.predict(data)
                    loss = self.se.forward(data)
                else:
                    output = self.model.predict(data)
                    loss = self.model(data)

                loss = loss.mean().data.cpu().numpy()
                loss_log.append(loss)

                if self.model.n_out == 1:
                    output = output.view(-1).data.cpu().numpy()
                else:
                    output = output.view(-1, self.model.n_out).data.cpu().numpy()

                pred.append(output)
                target.append(data["label"].view(-1).data.cpu().numpy())

        self.model.train()

        return loss_log, (
            np.vstack(target) if len(target[0].shape) == 2 else np.hstack(target),
            np.vstack(pred) if len(pred[0].shape) == 2 else np.hstack(pred),
        )

    def predict(self, dataloader: DataLoader, stage: str) -> np.ndarray:
        """Predict model.

        Args:
            dataloader: Torch dataloader.
            stage: Train, val or test.

        Returns:
            Prediction.

        """

        loss, (target, pred) = self.test(stage=stage, snap=self.is_snap, dataloader=dataloader)
        return pred
