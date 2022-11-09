"""Linear models based on Torch library."""

import logging
from copy import deepcopy
from typing import Callable, Optional, Sequence, Union

import cupy as cp
import numpy as np
import torch
from cupyx.scipy import sparse as sparse_cupy
from torch import nn, optim

from lightautoml.tasks.losses import TorchLossWrapper

logger = logging.getLogger(__name__)
ArrayOrSparseMatrix = Union[cp.ndarray, sparse_cupy.spmatrix]


def convert_cupy_scipy_sparse_to_torch_float(
    matrix: sparse_cupy.spmatrix, dev_id: int
) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse tensor (GPU version).

    Args:
        matrix: Matrix to convert.

    Returns:
        Matrix in torch.Tensor format.

   """
    matrix = sparse_cupy.coo_matrix(matrix, dtype=np.float32)
    cp_idx = cp.stack([matrix.row, matrix.col], axis=0).astype(cp.int64)
    idx = torch.as_tensor(cp_idx, device=f"cuda:{dev_id}")
    values = torch.as_tensor(matrix.data, device=f"cuda:{dev_id}")
    sparse_tensor = torch.sparse_coo_tensor(idx, values, size=matrix.shape)

    return sparse_tensor


class CatLinear(nn.Module):
    """Simple linear model to handle numeric and categorical features (GPU version)."""

    def __init__(
        self,
        numeric_size: int = 0,
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
    ):
        """
        Args:
            numeric_size: Number of numeric features.
            embed_sizes: Embedding sizes.
            output_size: Size of output layer.

        """
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(output_size).cuda())

        # add numeric if it is defined
        self.linear = None
        if numeric_size > 0:
            self.linear = nn.Linear(
                in_features=numeric_size, out_features=output_size, bias=False
            ).cuda()
            nn.init.zeros_(self.linear.weight)

        # add categories if it is defined
        self.cat_params = None
        if len(embed_sizes) > 0:
            self.cat_params = nn.Parameter(
                torch.zeros(embed_sizes.sum(), output_size).cuda()
            )
            self.register_buffer(
                "embed_idx",
                torch.LongTensor(embed_sizes).cumsum(dim=0)
                - torch.LongTensor(embed_sizes),
            )

    def forward(
        self,
        numbers: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
    ):
        """Forward-pass.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        """
        x = self.bias

        if self.linear is not None:
            x = x + self.linear(numbers)

        if self.cat_params is not None:
            x = x + self.cat_params[categories + self.embed_idx].sum(dim=1)

        return x


class CatLogisticRegression(CatLinear):
    """Realisation of torch-based logistic regression (GPU version)."""

    def __init__(
        self, numeric_size: int, embed_sizes: Sequence[int] = (), output_size: int = 1
    ):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        numbers: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
    ):
        """Forward-pass. Sigmoid func at the end of linear layer.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        """
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.sigmoid(x)

        return x


class CatRegression(CatLinear):
    """Realisation of torch-based linear regreession (GPU version)."""

    def __init__(
        self, numeric_size: int, embed_sizes: Sequence[int] = (), output_size: int = 1
    ):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)


class CatMulticlass(CatLinear):
    """Realisation of multi-class linear classifier (GPU version)."""

    def __init__(
        self, numeric_size: int, embed_sizes: Sequence[int] = (), output_size: int = 1
    ):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        numbers: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
    ):
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.softmax(x)

        return x


class TorchBasedLinearEstimator:
    """Linear model based on torch L-BFGS solver (GPU version).

    Accepts Numeric + Label Encoded categories or Numeric sparse input.
    """

    def __init__(
        self,
        data_size: int,
        categorical_idx: Sequence[int] = (),
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
        cs: Sequence[float] = (
            0.00001,
            0.00005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            7.0,
            10.0,
            20.0,
        ),
        max_iter: int = 1000,
        tol: float = 1e-5,
        early_stopping: int = 2,
        loss=Optional[Callable],
        metric=Optional[Callable],
    ):
        """
        Args:
            data_size: Not used.
            categorical_idx: Indices of categorical features.
            embed_sizes: Categorical embedding sizes.
            output_size: Size of output layer.
            cs: Regularization coefficients.
            max_iter: Maximum iterations of L-BFGS.
            tol: Tolerance for the stopping criteria.
            early_stopping: Maximum rounds without improving.
            loss: Loss function. Format: loss(preds, true) -> loss_arr, assume ```reduction='none'```.
            metric: Metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """

        self.data_size = data_size
        self.categorical_idx = categorical_idx
        self.embed_sizes = embed_sizes
        self.output_size = output_size

        assert all([x > 0 for x in cs]), "All Cs should be greater than 0"

        self.cs = cs
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.loss = loss  # loss(preds, true) -> loss_arr, assume reduction='none'
        self.metric = metric  # metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better)

    def _prepare_data(self, data: ArrayOrSparseMatrix, dev_id: int = 0):
        """Prepare data based on input type.

        Args:
            data: Data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        if sparse_cupy.issparse(data):
            return self._prepare_data_sparse(data, dev_id)

        return self._prepare_data_dense(data, dev_id)

    def _prepare_data_sparse(self, data: sparse_cupy.spmatrix, dev_id: int = 0):
        """Prepare sparse matrix.

        Only supports numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, `None`).

        """
        assert (
            len(self.categorical_idx["int"]) == 0
        ), "Support only numeric with sparse matrix"
        data = convert_cupy_scipy_sparse_to_torch_float(data, dev_id)
        return data, None

    def _prepare_data_dense(self, data, dev_id: int = 0):
        """Prepare dense matrix.

        Split categorical and numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """

        if 0 < len(self.categorical_idx["int"]) < data.shape[1]:

            data_cat = torch.as_tensor(
                data[:, self.categorical_idx["int"]].astype(cp.int32),
                device=f"cuda:{dev_id}",
            )

            data = torch.as_tensor(
                data[
                    :,
                    np.setdiff1d(np.arange(data.shape[1]), self.categorical_idx["int"]),
                ].astype(cp.float32),
                device=f"cuda:{dev_id}",
            )
            return data, data_cat

        elif len(self.categorical_idx["int"]) == 0:
            data = torch.as_tensor(data.astype(cp.float32), device=f"cuda:{dev_id}")
            return data, None

        else:
            data_cat = torch.as_tensor(data.astype(cp.int32), device=f"cuda:{dev_id}")
            return None, data_cat

    def _optimize(
        self,
        data: torch.Tensor,
        data_cat: Optional[torch.Tensor],
        y: torch.Tensor = None,
        weights: Optional[torch.Tensor] = None,
        c: float = 1.0,
    ):
        """Optimize single model.

        Args:
            data: Numeric data to train.
            data_cat: Categorical data to train.
            y: Target values.
            weights: Item weights.
            c: Regularization coefficient.

        """
        self.model.train()
        opt = optim.LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=self.max_iter,
            tolerance_change=self.tol,
            tolerance_grad=self.tol,
            line_search_fn="strong_wolfe",
        )

        # keep history
        results = []

        def closure():
            opt.zero_grad()
            output = self.model(data, data_cat)
            loss = self._loss_fn(y, output, weights, c).cuda()
            if loss.requires_grad:
                loss.backward()
            results.append(loss.item())
            return loss

        opt.step(closure)

    def _loss_fn(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        weights: Optional[torch.Tensor],
        c: float,
    ) -> torch.Tensor:
        """Weighted loss_fn wrapper.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            weights: Item weights.
            c: Regularization coefficients.

        Returns:
            Loss+Regularization value.

        """
        # weighted loss
        loss = self.loss(y_true, y_pred, sample_weight=weights)

        n = y_true.shape[0]
        if weights is not None:
            n = weights.sum()

        all_params = torch.cat(
            [y.view(-1) for (x, y) in self.model.named_parameters() if x != "bias"]
        )
        # print("AP device:", all_params.device)
        penalty = torch.norm(all_params, 2).pow(2) / 2 / n
        # print("Penalty device:", penalty.device)
        return loss + 0.5 / c * penalty

    def fit(
        self,
        data: cp.ndarray,
        y: cp.ndarray,
        weights: Optional[np.ndarray] = None,
        data_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        weights_val: Optional[np.ndarray] = None,
        dev_id: int = 0,
    ):
        """Fit method.

        Args:
            data: Data to train.
            y: Train target values.
            weights: Train items weights.
            data_val: Data to validate.
            y_val: Valid target values.
            weights_val: Validation item weights.

        Returns:
            self.

        """
        assert self.model is not None, "Model should be defined"

        data, data_cat = self._prepare_data(data, dev_id)

        if len(y.shape) == 1:
            y = y[:, cp.newaxis]
        y = torch.as_tensor(y.astype(cp.float32), device=f"cuda:{dev_id}")
        if weights is not None:
            weights = torch.as_tensor(
                weights.astype(cp.float32), device=f"cuda:{dev_id}"
            )
        if data_val is None and y_val is None:
            logger.warning(
                "Validation data should be defined. No validation will be performed and C = 1 will be used"
            )
            self._optimize(data, data_cat, y, weights, 1.0)

            return self
        data_val, data_val_cat = self._prepare_data(data_val, dev_id)
        best_score = -np.inf
        best_model = None
        es = 0
        for c in self.cs:
            self._optimize(data, data_cat, y, weights, c)
            val_pred = self._score(data_val, data_val_cat)
            score = self.metric(y_val, val_pred, weights_val)
            logger.info("Linear model: C = {0} score = {1}".format(c, score))
            if score > best_score:

                best_score = score
                best_model = deepcopy(self.model)
                es = 0
            else:

                es += 1

            if es >= self.early_stopping:
                break

            self.model = best_model

        return self

    def _score(self, data: cp.ndarray, data_cat: Optional[cp.ndarray]) -> cp.ndarray:
        """Get predicts to evaluate performance of model.

        Args:
            data: Numeric data.
            data_cat: Categorical data.

        Returns:
            Predicted target values.

        """
        preds = None
        with torch.set_grad_enabled(False):
            self.model.eval()
            preds = cp.asarray(self.model(data, data_cat))
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = cp.squeeze(preds)
        return preds

    def predict(self, data: cp.ndarray, dev_id: int = 0) -> cp.ndarray:
        """Inference phase.

        Args:
            data: Data to test.

        Returns:
            Predicted target values.

        """
        data, data_cat = self._prepare_data(data, dev_id)
        res = self._score(data, data_cat)
        return res


class TorchBasedLogisticRegression(TorchBasedLinearEstimator):
    """Linear binary classifier."""

    def __init__(
        self,
        data_size: int,
        categorical_idx: Sequence[int] = (),
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
        cs: Sequence[float] = (
            0.00001,
            0.00005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            7.0,
            10.0,
            20.0,
        ),
        max_iter: int = 1000,
        tol: float = 1e-4,
        early_stopping: int = 2,
        loss=Optional[Callable],
        metric=Optional[Callable],
    ):
        """
        Args:
            data_size: not used.
            categorical_idx: indices of categorical features.
            embed_sizes: categorical embedding sizes.
            output_size: size of output layer.
            cs: regularization coefficients.
            max_iter: maximum iterations of L-BFGS.
            tol: the tolerance for the stopping criteria.
            early_stopping: maximum rounds without improving.
            loss: loss function. Format: loss(preds, true) -> loss_arr, assume reduction='none'.
            metric: metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        if output_size == 1:
            _loss = nn.BCELoss
            _model = CatLogisticRegression
            self._binary = True
        else:
            _loss = nn.CrossEntropyLoss
            _model = CatMulticlass
            self._binary = False
        if loss is None:
            loss = TorchLossWrapper(_loss)
        super().__init__(
            data_size,
            categorical_idx,
            embed_sizes,
            output_size,
            cs,
            max_iter,
            tol,
            early_stopping,
            loss,
            metric,
        )
        self.model = _model(
            self.data_size - len(self.categorical_idx["int"]),
            self.embed_sizes,
            self.output_size,
        ).cuda()

    def predict(self, data: cp.ndarray, dev_id: int = 0) -> cp.ndarray:
        """Inference phase.

        Args:
            data: data to test.

        Returns:
            predicted target values.

        """
        pred = super().predict(data, dev_id)
        return pred


class TorchBasedLinearRegression(TorchBasedLinearEstimator):
    """Torch-based linear regressor optimized by L-BFGS."""

    def __init__(
        self,
        data_size: int,
        categorical_idx: Sequence[int] = (),
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
        cs: Sequence[float] = (
            0.00001,
            0.00005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            7.0,
            10.0,
            20.0,
        ),
        max_iter: int = 1000,
        tol: float = 1e-4,
        early_stopping: int = 2,
        loss=Optional[Callable],
        metric=Optional[Callable],
    ):
        """
        Args:
            data_size: used only for super function.
            categorical_idx: indices of categorical features.
            embed_sizes: categorical embedding sizes
            output_size: size of output layer.
            cs: regularization coefficients.
            max_iter: maximum iterations of L-BFGS.
            tol: the tolerance for the stopping criteria.
            early_stopping: maximum rounds without improving.
            loss: loss function. Format: loss(preds, true) -> loss_arr, assume reduction='none'.
            metric: metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        if loss is None:
            loss = TorchLossWrapper(nn.MSELoss)
        super().__init__(
            data_size,
            categorical_idx,
            embed_sizes,
            output_size,
            cs,
            max_iter,
            tol,
            early_stopping,
            loss,
            metric,
        )
        self.model = CatRegression(
            self.data_size - len(self.categorical_idx["int"]),
            self.embed_sizes,
            self.output_size,
        ).cuda()

    def predict(self, data: cp.ndarray, dev_id: int = 0) -> cp.ndarray:
        """Inference phase.

        Args:
            data: data to test.

        Returns:
            predicted target values.

        """
        return super().predict(data, dev_id)
