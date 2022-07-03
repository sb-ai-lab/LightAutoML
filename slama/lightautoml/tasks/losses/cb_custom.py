"""Custom metrics and loss functions for Catboost."""

from typing import Callable

import numpy as np


# TODO: Calc metrics on gpu slow down warning. Check it


class CBCustomMetric:
    """Metric wrapper class for CatBoost."""

    def __init__(self, metric: Callable, greater_is_better: bool = True, bw_func: Callable = None):
        """

        Args:
            metric: Callable metric.
            greater_is_better: Bool with metric direction.

        """
        self.metric = metric
        self.greater_is_better = greater_is_better
        self._bw_func = bw_func

    @staticmethod
    def get_final_error(error, weight):
        return error

    def is_max_optimal(self):
        return self.greater_is_better

    def evaluate(self, approxes, target, weight):
        raise NotImplementedError

    def _transform(self, approxes, target):
        target = np.array(target)
        pred = np.array(approxes[0])
        if self._bw_func is not None:
            target = self._bw_func(target)
            pred = self._bw_func(pred)

        return pred, target

    def _evaluate(self, approxes, target, weight):
        pred, target = self._transform(approxes, target)
        if weight is None:
            score = self.metric(target, pred)
            return score, 1
        else:
            try:
                score = self.metric(target, pred, sample_weight=np.array(weight))
            except TypeError:
                score = self.metric(target, pred)
            return score, weight


class CBRegressionMetric(CBCustomMetric):
    """Regression metric wrapper for CatBoost."""

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(approxes[0]) == len(target)
        assert weight is None or len(target) == len(weight)

        return self._evaluate(approxes, target, weight)


class CBClassificationMetric(CBCustomMetric):
    """Classification metric wrapper for CatBoost."""

    def __init__(
        self,
        metric: Callable,
        greater_is_better: bool,
        bw_func: Callable = None,
        use_proba: bool = True,
    ):
        super(CBClassificationMetric, self).__init__(metric, greater_is_better, bw_func)
        self.use_proba = use_proba

    def _transform(self, approxes, target):
        target = np.array(target)
        pred = np.array(approxes[0])
        if self._bw_func is not None:
            pred = self._bw_func(pred)

        pred = np.exp(pred)
        pred = pred / (1 + pred)
        if self.use_proba:
            return pred, target
        else:
            return np.round(pred), target

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(approxes[0]) == len(target)
        assert weight is None or len(target) == len(weight)

        return self._evaluate(approxes, target, weight)


class CBMulticlassMetric(CBCustomMetric):
    """Multiclassification metric wrapper for CatBoost."""

    def __init__(
        self,
        metric: Callable,
        greater_is_better: bool,
        bw_func: Callable = None,
        use_proba: bool = True,
    ):
        super().__init__(metric, greater_is_better, bw_func)
        self.use_proba = use_proba

    def _transform(self, approxes, target):
        target = np.array(target)
        pred = np.array(approxes)
        if self._bw_func is not None:
            pred = self._bw_func(pred)

        pred = np.exp(approxes)
        pred = pred / (1 + pred)
        if self.use_proba:
            return pred, target
        else:
            return np.argmax(pred, axis=0), target

    def evaluate(self, approxes, target, weight):
        assert len(approxes[0]) == len(target)
        assert weight is None or len(target) == len(weight)

        return self._evaluate(approxes, target, weight)
