"""AutoMLWhitebox for tabular datasets."""

import warnings

from copy import copy
from copy import deepcopy
from typing import Optional
from typing import Tuple
from typing import Union

import autowoe
import numpy as np

from pandas import DataFrame

from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..validation.base import TrainValidIterator
from .base import TabularMLAlgo


WbModel = Union[autowoe.AutoWoE, autowoe.ReportDeco]


class WbMLAlgo(TabularMLAlgo):
    """WhiteBox - scorecard model.

    https://github.com/sberbank-ai-lab/AutoMLWhitebox

    default_params:

       - monotonic: bool
           Global condition for monotonic constraints.
           If ``True``, then only monotonic binnings will be built.
           You can pass values to the ``.fit``
           method that change this condition
           separately for each feature.
       - max_bin_count: int
           Global limit for the number of bins. Can be specified for every
           feature in .fit
       - select_type: ``None`` or ``int``
           The type to specify the primary feature selection.
           If the type is an integer, then we select the number
           of features indicated by this number
           (with the best `feature_importance`).
           If the value is ``None``, we leave only features
           with ``feature_importance`` greater than ``0``.
       - pearson_th:  0 < pearson_th < 1
           Threshold for feature selection by correlation. All features with
           the absolute value of correlation coefficient greater then
           pearson_th will be discarded.
       - auc_th: .5 < auc_th < 1
           Threshold for feature selection by one-dimensional AUC.
           WoE with AUC < auc_th will be discarded.
       - vif_th: vif_th > 0
           Threshold for feature selection by VIF. Features with VIF > vif_th
           are iteratively discarded one by one, then VIF is recalculated
           until all VIFs are less than vif_th.
       - imp_th: real >= 0
           Threshold for feature selection by feature importance
       - th_const:
           Threshold, which determines that the feature is constant.
           If the number of valid values is greater than the threshold, then
           the column is not constant. For float, the number of
           valid values will be calculated as the sample size * th_const
       - force_single_split: bool
           In the tree parameters, you can set the minimum number of
           observations in the leaf. Thus, for some features,
           splitting for 2 beans at least will be impossible.
           If you specify that ``force_single_split = True``,
           it means that 1 split will be created for the feature,
           if the minimum bin size is greater than th_const.
       - th_nan: int >= 0
           Threshold, which determines that WoE values are calculated to NaN.
       - th_cat: int >= 0
           Threshold, which determines which categories are small.
       - woe_diff_th: float = 0.01
           The option to merge NaNs and rare categories with another bin,
           if the difference in WoE is less than woe_diff_th.
       - min_bin_size: int > 1, 0 < float < 1
           Minimum bin size when splitting.
       - min_bin_mults: list of floats > 1
           If minimum bin size is specified, you can specify a list to check
           if large values work better, for example: [2, 4].
       - min_gains_to_split: list of floats >= 0
           min_gain_to_split values that will be
           iterated to find the best split.
       - auc_tol: 1e-5 <= auc_tol <=1e-2
           AUC tolerance. You can lower the auc_tol value from the maximum
           to make the model simpler.
       - cat_alpha: float > 0
           Regularizer for category encoding.
       - cat_merge_to: str
           The way of WoE values filling in the test sample for categories
           that are not in the training sample.
           Values - 'to_nan', 'to_woe_0', 'to_maxfreq', 'to_maxp', 'to_minp'
       - nan_merge_to: str
           The way of WoE values filling on the test sample for real NaNs,
           if they are not included in their group.
           Values - 'to_woe_0', 'to_maxfreq', 'to_maxp', 'to_minp'
       - oof_woe: bool
           Use OOF or standard encoding for WOE.
       - n_folds: int
           Number of folds for feature selection / encoding, etc.
       - n_jobs: int > 0
           Number of CPU cores to run in parallel.
       - l1_base_step: real > 0
           Grid size in l1 regularization
       - l1_exp_step: real > 1
           Grid scale in l1 regularization
       - population_size: None, int > 0
           Feature selection type in the selector.
           If the value is ``None`` then L1 boost is used.
           If ``int`` is specified, then a standard step will be used for
           the number of random subsamples indicated by this value.
           Can be generalized to genetic algorithm.
       - feature_groups_count: int > 0
           The number of groups in the genetic algorithm.
           Its effect is visible only when population_size > 0
       - imp_type: str
           Feature importances type. Feature_imp and perm_imp are available.
           It is used to sort the features at the first and at the final
           stage of feature selection.
       - regularized_refit: bool
           Use regularization at the time of model refit. Otherwise, we have
           a statistical model.
       - p_val: 0 < p_val <= 1
           When training a statistical model, do backward selection
           until all p-values of the model's coefficient are
       - verbose: int 0-3
           Verbosity level

    freeze_defaults:
        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "WhiteBox"

    _default_params = {
        "monotonic": False,
        "max_bin_count": 5,
        "select_type": None,
        "pearson_th": 0.9,
        "auc_th": 0.505,
        "vif_th": 10.0,
        "imp_th": 0,
        "th_const": 32,
        "force_single_split": True,
        "th_nan": 0.01,
        "th_cat": 0.005,
        "woe_diff_th": 0.01,
        "min_bin_size": 0.01,
        "cat_alpha": 100,
        "cat_merge_to": "to_woe_0",
        "nan_merge_to": "to_woe_0",
        "oof_woe": True,
        "n_folds": 6,
        "n_jobs": 4,
        "l1_grid_size": 20,
        "l1_exp_scale": 6,
        "imp_type": "feature_imp",
        "regularized_refit": False,
        "p_val": 0.05,
        "report": False,
    }

    _report_on_inference = False

    def _infer_params(self) -> Tuple[dict, bool, dict]:

        params = deepcopy(self.params)
        report = params.pop("report")
        fit_params = params.pop("fit_params")
        self._report_on_inference = report
        return params, report, fit_params

    def fit_predict(self, train_valid_iterator: TrainValidIterator, **kwargs) -> NumpyDataset:

        self._dataset_fit_params = kwargs

        return super().fit_predict(train_valid_iterator)

    def _include_target(self, dataset: PandasDataset, include_group: bool = False) -> Tuple[DataFrame, Optional[str]]:

        df = dataset.data.copy()
        if dataset.target is not None:
            df["__TARGET__"], _ = self.task.losses["lgb"].fw_func(dataset.target.values, None)
        group_kf = None

        if include_group and dataset.group is not None:
            assert "__GROUP__" not in dataset.features, "__GROUP__ is not valid column name for WhiteBox"
            df["__GROUP__"] = dataset.group.values
            group_kf = "__GROUP__"

        return df, group_kf

    def fit_predict_single_fold(self, train: PandasDataset, valid: PandasDataset) -> Tuple[WbModel, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        params, report, fit_params = self._infer_params()

        assert train.task.name == "binary", "Only binary task is supported"
        assert "__TARGET__" not in train.features, "__TARGET__ is not valid column name for WhiteBox"
        if train.weights is not None:
            warnings.warn("Weights are ignored at the moment", UserWarning, stacklevel=2)

        train_df, group_kf = self._include_target(train, True)

        roles = train.roles
        mapping = {"Category": "cat", "Numeric": "real"}
        features_type = {x: mapping[roles[x].name] for x in roles}

        valid_df = None
        if train is not valid:
            valid_df, _ = self._include_target(valid, False)

        model = autowoe.AutoWoE(**params)

        if report:
            model = autowoe.ReportDeco(model)

        kwargs = copy(self._dataset_fit_params)
        kwargs["validation"] = valid_df
        kwargs = {**kwargs, **fit_params}

        model.fit(train_df, target_name="__TARGET__", group_kf=group_kf, features_type=features_type, **kwargs)

        if train is valid:
            valid_df = train_df

        val_pred = model.predict_proba(valid_df)
        val_pred = self.task.losses["lgb"].bw_func(val_pred)

        return model, val_pred

    def predict_single_fold(self, model: WbModel, dataset: PandasDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: WhiteBox model
            dataset: Test dataset.

        Return:
            Predicted target values.

        """
        args = []
        if self.params["report"]:
            args = [self._report_on_inference]

        df, _ = self._include_target(dataset, False)
        pred = self.task.losses["lgb"].bw_func(model.predict_proba(df, *args))

        return pred

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with ImportanceEstimator.

        Args:
            train_valid: classic cv iterator.

        """
        self.fit_predict(train_valid)

    def predict(self, dataset: PandasDataset, report: bool = False) -> NumpyDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset.
            report: Flag to generate report.

        Returns:
            Dataset with predictions.

        """
        self._report_on_inference = report
        return super().predict(dataset)
