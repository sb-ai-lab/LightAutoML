# Standard python libraries
import logging

from copy import deepcopy


logging.basicConfig(format="[%(asctime)s] (%(levelname)s): %(message)s", level=logging.INFO)

# Installed libraries
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose

# Imports from our package
from ...automl.base import AutoML
from ...automl.blend import WeightedBlender
from ...dataset.roles import DatetimeRole
from ...ml_algo.boost_cb import BoostCB
from ...ml_algo.linear_sklearn import LinearLBFGS
from ...ml_algo.random_forest import RandomForestSklearn
from ...pipelines.features.lgb_pipeline import LGBSeqSimpleFeatures
from ...pipelines.features.linear_pipeline import LinearTrendFeatures
from ...pipelines.ml.base import MLPipeline
from ...reader.base import DictToPandasSeqReader
from ...tasks import Task


class TrendModel:

    _available_trend_types = ["decompose", "decompose_STL", "linear", "rolling"]
    default_params = {
        "trend": True,
        "train_on_trend": True,
        "trend_type": "decompose",
        "trend_size": 7,
        "decompose_period": 30,
        "detect_step_quantile": 0.01,
        "detect_step_window": 7,
        "detect_step_threshold": 0.7,
        "rolling_size": 7,
        "verbose": 0,
    }

    def __init__(self, params=None):
        self.params = deepcopy(self.default_params)
        if params is not None:
            self.params.update(params)
        assert self.params["trend_type"] in self._available_trend_types

    def _detect_step(self, x):
        x_min, x_max = tuple(
            np.quantile(x, [self.params["detect_step_quantile"], 1 - self.params["detect_step_quantile"]])
        )
        x_range = x_max - x_min
        window = self.params["detect_step_window"]
        diff = np.zeros(len(x) - 2 * window)
        for i in range(len(x) - 2 * window):
            diff[i] = np.median(x[i + window : i + 2 * window]) - np.median(x[i : i + window])
        diff = np.abs(diff) / x_range
        diff = np.concatenate((diff[0] * np.ones(window), diff, diff[-1] * np.ones(window)))
        return np.any(diff > self.params["detect_step_threshold"])

    def _get_rolling_median(self, train_data, roles):
        median = train_data[roles["target"]].rolling(self.params["rolling_size"]).apply(np.median)
        return median.fillna(median[~median.isna()].values[0]).values

    def _detect_no_trend(self, x):
        pass

    def _estimate_trend(self, train_data, roles):
        if self.params["trend_type"] == "decompose":
            return seasonal_decompose(
                train_data[roles["target"]].values,
                model="additive",
                period=self.params["decompose_period"],
                extrapolate_trend="freq",
            ).trend
        elif self.params["trend_type"] == "decompose_STL":
            return STL(train_data[roles["target"]].values, period=self.params["decompose_period"]).fit().trend
        elif self.params["trend_type"] == "rolling":
            return self._get_rolling_median(train_data, roles)

    def fit_predict(self, train_data, roles):
        """Fit and predict data.

        if self._detect_no_trend(train_data[roles['target']]):
            self.params['trend'] = False

        # noqa: DAR101
        # noqa: DAR201

        """
        self.roles = roles
        if not self.params["trend"]:
            return np.zeros(len(train_data))
        if self._detect_step(train_data[roles["target"]]):
            self.params["trend_type"] = "rolling"

        task_trend = Task("reg", greater_is_better=False, metric="mae", loss="mae")
        reader_trend = DictToPandasSeqReader(task=task_trend, cv=2, seq_params={})
        reader_trend.fit_read({"plain": train_data, "seq": None}, roles=roles)
        timerole = [key for key, value in reader_trend._roles.items() if isinstance(value, DatetimeRole)][0]

        feats_trend = LinearTrendFeatures()
        model_trend = LinearLBFGS()
        pipeline_trend = MLPipeline(
            [model_trend], pre_selection=None, features_pipeline=feats_trend, post_selection=None
        )
        self.automl_trend = AutoML(reader_trend, [[pipeline_trend]], skip_conn=False)

        if self.params["trend_type"] in ["decompose", "decompose_STL", "rolling"]:
            trend = self._estimate_trend(train_data, roles)
            if self.params["train_on_trend"]:
                trend_data = train_data[[roles["target"], timerole]].copy()
                trend_data.drop(roles["target"], axis=1, inplace=True)
                trend_data["trend"] = trend
                roles = {"target": "trend"}
                _ = self.automl_trend.fit_predict(
                    {"plain": trend_data.iloc[-self.params["trend_size"] :], "seq": None}, roles=roles
                )
            else:
                _ = self.automl_trend.fit_predict(
                    {"plain": train_data[[roles["target"], timerole]].iloc[-self.params["trend_size"] :], "seq": None},
                    roles=roles,
                )

        elif self.params["trend_type"] == "linear":
            _ = self.automl_trend.fit_predict(
                {"plain": train_data[[roles["target"], timerole]], "seq": None}, roles=roles
            )
            trend = self.automl_trend.predict({"plain": train_data, "seq": None}).data[:, 0]
        return trend

    def predict(self, data, future_time):
        MIN_PREDICT_HISTORY = 5 * self.params["trend_size"]
        if not self.params["trend"]:
            return np.zeros(len(data)), np.zeros(len(future_time))
        if self.params["trend_type"] == "linear" or len(data) < MIN_PREDICT_HISTORY:
            trend = self.automl_trend.predict({"plain": data, "seq": None}).data[:, 0]
        else:
            trend = self._estimate_trend(data, self.roles)
        pred_trend = self.automl_trend.predict({"plain": future_time, "seq": None}).data[:, 0]
        return trend, pred_trend


class AutoTS:

    default_trend_params = {
        "trend": True,
        "train_on_trend": True,
        "trend_type": "decompose",  # 'decompose', 'decompose_STL', 'linear' or 'rolling'
        "trend_size": 7,
        "decompose_period": 30,
        "detect_step_quantile": 0.01,
        "detect_step_window": 7,
        "detect_step_threshold": 0.7,
        "rolling_size": 7,
        "verbose": 0,
    }

    default_transformers_params = {  # True (then set default value), False, int, list or np.array
        "lag_features": True,
        "lag_time_features": True,
        "diff_features": False,
    }

    @property
    def n_target(self):
        """Get length of future prediction.

        Returns:
            length
        """
        return self.seq_params["seq0"]["params"]["n_target"]

    @property
    def n_history(self):
        """Get length of history used for feature generation.

        Returns:
            length
        """
        return self.seq_params["seq0"]["params"]["history"]

    @property
    def datetime_key(self):
        """Get name of datetime index column

        Returns:
            column name
        """
        return (
            self.TM.automl_trend.levels[0][0]
            .features_pipeline._pipeline.transformer_list[0]
            .transformer_list[0]
            .keys[0]
        )

    def __init__(self, task, seq_params=None, trend_params=None, transformers_params=None):
        self.task = task
        self.task_trend = Task("reg", greater_is_better=False, metric="mae", loss="mae")
        if seq_params is None:
            self.seq_params = {
                "seq0": {
                    "case": "next_values",
                    "params": {"n_target": 7, "history": 7, "step": 1, "from_last": True, "test_last": True},
                },
            }
        else:
            self.seq_params = seq_params
        self.test_last = self.seq_params["seq0"]["params"]["test_last"]

        # Trend params
        self.trend_params = deepcopy(self.default_trend_params)
        if trend_params is not None:
            self.trend_params.update(trend_params)
        self.TM = TrendModel(params=self.trend_params)

        # Transformers params
        self.transformers_params = deepcopy(self.default_transformers_params)
        if transformers_params is not None:
            self.transformers_params.update(transformers_params)

        # default params if they have been stated as boolean True
        if isinstance(self.transformers_params["lag_features"], bool) and self.transformers_params["lag_features"]:
            self.transformers_params["lag_features"] = 30
        if (
            isinstance(self.transformers_params["lag_time_features"], bool)
            and self.transformers_params["lag_time_features"]
        ):
            self.transformers_params["lag_time_features"] = 7
        if isinstance(self.transformers_params["diff_features"], bool) and self.transformers_params["diff_features"]:
            self.transformers_params["diff_features"] = 7

    def fit_predict(self, train_data, roles, verbose=0):
        self.roles = roles
        train_trend = self.TM.fit_predict(train_data, roles)

        if hasattr(self.TM, "automl_trend"):
            self.datetime_step = (
                pd.to_datetime(train_data[self.datetime_key])[1] - pd.to_datetime(train_data[self.datetime_key])[0]
            )
        # fit main
        train_detrend = train_data.copy()
        train_detrend.loc[:, roles["target"]] = train_detrend.loc[:, roles["target"]] - train_trend

        reader_seq = DictToPandasSeqReader(task=self.task, cv=2, seq_params=self.seq_params)
        feats_seq = LGBSeqSimpleFeatures(fill_na=True, scaler=True, transformers_params=self.transformers_params)
        model = RandomForestSklearn(default_params={"verbose": 0})
        # model2 = LinearLBFGS(default_params={'cs': [1]})
        model2 = LinearLBFGS()

        model3 = BoostCB()
        pipeline_lvl1 = MLPipeline([model], pre_selection=None, features_pipeline=feats_seq, post_selection=None)
        pipeline2_lvl1 = MLPipeline([model2], pre_selection=None, features_pipeline=feats_seq, post_selection=None)
        pipeline3_lvl1 = MLPipeline([model3], pre_selection=None, features_pipeline=feats_seq, post_selection=None)
        self.automl_seq = AutoML(
            reader_seq, [[pipeline_lvl1, pipeline2_lvl1, pipeline3_lvl1]], skip_conn=False, blender=WeightedBlender()
        )

        oof_pred_seq = self.automl_seq.fit_predict({"seq": {"seq0": train_detrend}}, roles=roles, verbose=verbose)
        return oof_pred_seq, train_trend

    def predict(self, data, return_raw=False):
        test_idx = None
        if self.trend_params["trend"] is True:
            last_datetime = pd.to_datetime(data[self.datetime_key]).values[-1]
            vals = [last_datetime + (i + 1) * self.datetime_step for i in range(self.n_target)]
            if not self.test_last:
                vals = data[self.datetime_key].tolist() + vals
            test_data = pd.DataFrame(vals, columns=[self.datetime_key])
            if not self.test_last:
                test_idx = self.automl_seq.reader.ti["seq0"].create_target(test_data, plain_data=None)
            trend, test_pred_trend = self.TM.predict(data, test_data)
        else:
            test_pred_trend = np.zeros(self.n_target)
            trend = np.zeros(len(data))

        detrend = data.copy()
        detrend.loc[:, self.roles["target"]] = detrend.loc[:, self.roles["target"]] - trend
        test_pred_detrend = self.automl_seq.predict({"seq": {"seq0": detrend}})
        if return_raw:
            return test_pred_detrend

        if test_pred_detrend.data.shape[0] == 1:
            final_pred = test_pred_trend + test_pred_detrend.data.flatten()
        else:
            if (test_idx is not None) and (not self.test_last):
                test_pred_trend = test_pred_trend[test_idx]
            final_pred = test_pred_trend + test_pred_detrend.data
        return final_pred, test_pred_trend
