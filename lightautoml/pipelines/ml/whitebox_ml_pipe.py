"""Whitebox MLPipeline."""

import warnings

from typing import Tuple
from typing import Union
from typing import cast

from ...dataset.np_pd_dataset import NumpyDataset
from ...dataset.np_pd_dataset import PandasDataset
from ...ml_algo.tuning.base import ParamsTuner
from ...ml_algo.whitebox import WbMLAlgo
from ...validation.base import DummyIterator
from ...validation.base import TrainValidIterator
from ..features.wb_pipeline import WBFeatures
from ..selection.base import EmptySelector
from .base import MLPipeline


TunedWB = Union[WbMLAlgo, Tuple[WbMLAlgo, ParamsTuner]]


class WBPipeline(MLPipeline):
    """Special pipeline to handle WhiteBox model."""

    @property
    def whitebox(self) -> WbMLAlgo:
        if len(self.ml_algos[0].models) > 1:
            warnings.warn("More than 1 whitebox model is fitted during cross validation. Only first is returned")

        return self.ml_algos[0].models[0]

    def __init__(self, whitebox: TunedWB):
        """Create WhiteBox MLPipeline.

        Args:
            whitebox: WhiteBox model.

        """
        super().__init__([whitebox], True, features_pipeline=WBFeatures())
        self._used_features = None

    def fit_predict(self, train_valid: TrainValidIterator) -> NumpyDataset:
        """Fit WhiteBox.

        Args:
            train_valid: Classic cv-iterator.

        Returns:
            Dataset.

        """
        _subsamp_to_refit = train_valid.train[:5]
        val_pred = super().fit_predict(train_valid)
        self._prune_pipelines(_subsamp_to_refit)

        return cast(NumpyDataset, val_pred)

    def predict(self, dataset: PandasDataset, report: bool = False) -> NumpyDataset:
        """Predict WhiteBox.

        Additional report param stands for WhiteBox report generation.

        Args:
            dataset: Dataset of text features.
            report: Flag if generate report.

        Returns:
            Dataset.

        """
        dataset = self.features_pipeline.transform(dataset)
        args = []
        if self.ml_algos[0].params["report"]:
            args = [report]
        pred = self.ml_algos[0].predict(dataset, *args)

        return pred

    def _prune_pipelines(self, subsamp: PandasDataset):
        # upd used features attribute from list of whiteboxes
        feats_from_wb = set.union(*[set(list(x.features_fit.index)) for x in self.ml_algos[0].models])
        # cols wo prefix - numerics and categories
        raw_columns = list(set(subsamp.features).intersection(feats_from_wb))
        diff_cols = list(set(feats_from_wb).difference(subsamp.features))

        seasons = ["__".join(x.split("__")[1:]) for x in diff_cols if x.startswith("season_")]

        base_diff = [x.split("__") for x in diff_cols if x.startswith("basediff_")]
        base_diff = [("_".join(x[0].split("_")[1:]), "__".join(x[1:])) for x in base_diff]
        base_dates, compare_dates = [x[0] for x in base_diff], [x[1] for x in base_diff]
        dates = list(set(base_dates + compare_dates + seasons))

        raw_columns.extend(dates)

        subsamp = subsamp[:, raw_columns]
        self.features_pipeline = WBFeatures()

        self.pre_selection = EmptySelector()
        self.post_selection = EmptySelector()

        train_valid = DummyIterator(subsamp)
        train_valid = train_valid.apply_selector(self.pre_selection)
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)
        train_valid.apply_selector(self.post_selection)
