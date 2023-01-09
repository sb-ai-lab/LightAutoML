"""Wrapped pboost for tabular datasets."""

import logging

import os

import pandas as pd
import numpy as np
from py_boost import TLPredictor

from lightautoml_gpu.dataset.np_pd_dataset import NumpyDataset

from lightautoml_gpu.pipelines.selection.base import ImportanceEstimator
from .base import TabularMLAlgo

logger = logging.getLogger(__name__)


class PBPredictor(TabularMLAlgo, ImportanceEstimator):
    """Boosting using PyBoost library for cpu inference.
    """

    _name: str = "PB"
    _folder_name = "treelite"

    def __getstate__(self):
        if os.path.exists(self._folder_name + "/" + self._name):
            logger.info("Folder \"{}\" already exists".format(self._folder_name + "/" + self._name))
            i = 1
            while os.path.exists(self._folder_name + "/" + self._name + "_" + str(i)):
                i+=1
            self._name = self._name + "_" + str(i)
            logger.info("Saving py-boost model to \"{}\"".format(self._folder_name + "/" + self._name)) 
        for i in range(len(self.models)):
            self.models[i].dump("./" + self._folder_name + "/" + self._name + "/" + str(i))
        self.__dict__.pop("models")
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        models = []
        names = os.listdir("./" + self._folder_name + "/" + self._name)
        for name in names:
            models.append(TLPredictor.load("./" + self._folder_name + "/" + self._name + "/" + name))
        self.__dict__["models"] = models

    def predict_single_fold(
        self, model: TLPredictor, dataset: NumpyDataset
    ) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: TLPredictor object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        dataset = dataset.to_numpy()
        pred = model.predict(dataset.data)
        return pred

    def get_features_score(self) -> pd.Series:
        """Computes feature importance as mean values of feature importance provided by pyboost per all models.

        Returns:
            Series with feature importances.

        """

        imp = 0
        for model in self.models:
            imp = imp + model.get_feature_importance()

        imp = imp / len(self.models)

        return pd.Series(imp, index=self.features).sort_values(ascending=False)
