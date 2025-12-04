"""ICL methods for tabular datasets."""


from ..dataset.np_pd_dataset import PandasDataset, NumpyDataset, CSRSparseDataset
import numpy as np
import pandas as pd
import warnings
from tabicl import TabICLClassifier
from .base import TabularMLAlgo
from sklearn.impute import SimpleImputer
import torch
import logging

logger = logging.getLogger(__name__)


class TabICL(TabularMLAlgo):
    """TabICL classifier.

    Paper: https://arxiv.org/pdf/2502.05564
    """

    _name: str = "TabICL"

    def __init__(self, **kwargs) -> None:
        """Initialize TabICL classifier.

        Args:
            device: Device to use.
        """
        super().__init__()

        if not torch.cuda.is_available():
            logger.info("CUDA is not available, using CPU.")
            kwargs["device"] = "cpu"

        kwargs.pop("freeze_defaults", None)

        self.clf = TabICLClassifier(**kwargs)
        self.imputer = None
        self.is_multiclass_task = False
        self.imputer = SimpleImputer(strategy="median")

    def _fit(self, X_train, y_train):
        """Fit TabICL classifier.

        Args:
            X_train: Train data.
            y_train: Train labels.
        """
        X_train = self.imputer.fit_transform(X_train)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.clf.fit(X_train, y_train)
        if len(np.unique(y_train)) > 2:
            self.is_multiclass_task = True

    def _predict(self, X):
        """Predict with TabICL classifier.

        Args:
            X: Data to predict.

        Returns:
            Predictions.
        """
        if self.imputer is not None:
            X = self.imputer.transform(X)
        assert not np.isnan(X).any().any(), (self.imputer is not None, X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            preds = self.clf.predict_proba(X)

        return preds[:, None if self.is_multiclass_task else 1]

    def fit_predict_single_fold(self, train, val, **kwargs):
        """Fit and predict with TabICL classifier.

        Args:
            train: Train data.
            val: Validation data.
            **kwargs: Additional arguments.

        Returns:
            Model and predictions.
        """
        if isinstance(train, PandasDataset):
            train = train.to_numpy()
        if isinstance(val, PandasDataset):
            val = val.to_numpy()

        self._fit(train.data, train.target)

        return self.clf, self._predict(val.data)

    def predict_single_fold(self, model, data):
        """Predict with TabICL classifier.

        Args:
            model: TabICL classifier.
            data: Data to predict.

        Returns:
            Predictions.
        """
        if isinstance(data, (NumpyDataset, CSRSparseDataset, PandasDataset)):
            data = data.data

        if isinstance(data, pd.DataFrame):
            data = data.values

        if self.imputer is not None:
            data = self.imputer.transform(data)
        assert not np.isnan(data).any().any()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            preds = model.predict_proba(data)

        return preds[:, None if self.is_multiclass_task else 1]

    def fit(self, train_valid):
        """Fit TabICL classifier.

        Args:
            train_valid: Train and validation iterator.

        Returns:
            Predictions.
        """
        self.fit_predict(train_valid)
