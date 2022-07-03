from typing import Optional, Union, cast

from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator, \
    Evaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.pandas.functions import pandas_udf

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.utils import SparkDataFrame
from lightautoml.spark.tasks.losses.base import SparkLoss
from lightautoml.tasks import Task as LAMATask
from lightautoml.tasks.base import LAMLMetric, _default_losses

import pandas as pd
import numpy as np

DEFAULT_PREDICTION_COL_NAME = "prediction"
DEFAULT_TARGET_COL_NAME = "target"
DEFAULT_PROBABILITY_COL_NAME = "probability"


def argmax_in_vector(vec: pd.Series) -> pd.Series:
    vec.transform()


class SparkMetric(LAMLMetric):
    """
    Spark version of metric function that implements function assessing prediction error.
    """

    def __init__(
        self,
        name: str,
        metric_name: str,
        target_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
        greater_is_better: bool = True,
    ):
        """

        Args:
            metric: Specifies metric. Format:
                ``func(y_true, y_false, Optional[sample_weight], **kwargs)`` -> `float`.
            name: Name of metric.
            greater_is_better: Whether or not higher metric value is better.
            one_dim: `True` for single class, False for multiclass.
            weighted: Weights of classes.
            **kwargs: Other parameters for metric.

        """
        self._name = name
        self._metric_name = metric_name
        self._target_col = target_col
        self._prediction_col = prediction_col
        self.greater_is_better = greater_is_better

    def __call__(self, dataset: Union[SparkDataset, SparkDataFrame], dropna: bool = False):
        sdf: SparkDataFrame = dataset.data if isinstance(dataset, SparkDataset) else dataset

        if self._target_col is not None and self._prediction_col is not None:
            assert self._target_col in sdf.columns and self._prediction_col in sdf.columns
            prediction_column = self._prediction_col
            target_column = self._target_col
        elif isinstance(dataset, SparkDataset):
            if len(dataset.features) == 1:
                prediction_column = dataset.features[0]
            else:
                prediction_column = next(c for c in dataset.data.columns if c.startswith('prediction'))
            target_column = dataset.target_column
        else:
            sdf = cast(SparkDataFrame, dataset)
            assert DEFAULT_PREDICTION_COL_NAME in sdf.columns and DEFAULT_TARGET_COL_NAME in sdf.columns
            prediction_column = DEFAULT_PREDICTION_COL_NAME
            target_column = DEFAULT_TARGET_COL_NAME

        sdf = sdf.dropna(subset=[prediction_column, target_column]) if dropna else sdf

        if self._name == "binary":
            evaluator = BinaryClassificationEvaluator(rawPredictionCol=prediction_column)
        elif self._name == "reg":
            evaluator = RegressionEvaluator(predictionCol=prediction_column)
        else:
            temp_pred_col = 'multiclass_temp_prediction'
            evaluator = MulticlassClassificationEvaluator(probabilityCol=prediction_column, predictionCol=temp_pred_col)

            @pandas_udf('double')
            def argmax(vec: pd.Series) -> pd.Series:
                return vec.transform(lambda x: np.argmax(x))

            sdf = sdf.select('*', argmax(vector_to_array(prediction_column)).alias(temp_pred_col))

        evaluator = evaluator.setMetricName(self._metric_name).setLabelCol(target_column)

        score = evaluator.evaluate(sdf)
        sign = 2 * float(self.greater_is_better) - 1
        return score * sign

    @property
    def name(self) -> str:
        return self._name


class SparkTask(LAMATask):
    """
    Specify task (binary classification, multiclass classification, regression), metrics, losses.
    """

    _default_metrics = {"binary": "auc", "reg": "mse", "multiclass": "crossentropy"}

    _supported_metrics ={
        "binary": {
            "auc": "areaUnderROC",
            "aupr": "areaUnderPR"
        },
        "reg": {
            "r2": "rmse",
            "mse": "mse",
            "mae": "mae",
        },
        "multiclass": {
            "crossentropy": "logLoss",
            "accuracy": "accuracy",
            "f1_micro": "f1",
            "f1_weighted": "weightedFMeasure"
        }
    }

    def __init__(
        self,
        name: str,
        loss: Optional[str] = None,
        metric: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
    ):

        super().__init__(name, loss, None, metric, None, greater_is_better)

        if metric is None:
            metric = self._default_metrics[name]

        # add losses
        # if None - infer from task
        self.losses = {}
        if loss is None:
            loss = _default_losses[self.name]
        # SparkLoss actualy does nothing, but it is there
        # to male TabularAutoML work
        self.losses = {'lgb': SparkLoss(),'linear_l2': SparkLoss()}

        assert metric in self._supported_metrics[self.name], \
            f"Unsupported metric {metric} for task {self.name}." \
            f"The following metrics are supported: {list(self._supported_metrics[self.name].keys())}"

        self.metric_name = metric

    def get_dataset_metric(self) -> LAMLMetric:
        """ Obtains a function to calculate the metric on a dataset. """
        spark_metric_name = self._supported_metrics[self.name][self.metric_name]
        return SparkMetric(self.name, metric_name=spark_metric_name, greater_is_better=self.greater_is_better)
