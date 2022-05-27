"""Linear models for tabular datasets."""

import logging
from copy import copy
from typing import Tuple, Optional, List
from typing import Union

import numpy as np
from pyspark.ml import Pipeline, Transformer, PipelineModel, Estimator
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import functions as F

from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, AveragingTransformer
from lightautoml.spark.validation.base import SparkBaseTrainValidIterator
from ..dataset.base import SparkDataset
from ..transformers.base import DropColumnsTransformer
from ..utils import DebugTransformer, SparkDataFrame
from ...utils.timer import TaskTimer

logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, LinearRegression]
LinearEstimatorModel = Union[LogisticRegressionModel, LinearRegressionModel]


class SparkLinearLBFGS(SparkTabularMLAlgo):
    """LBFGS L2 regression based on Spark MLlib.


    default_params:

        - tol: The tolerance for the stopping criteria.
        - maxIter: Maximum iterations of L-BFGS.
        - aggregationDepth: Param for suggested depth for treeAggregate.
        - elasticNetParam: Elastic net parameter.
        - regParam: Regularization parameter.
        - early_stopping: Maximum rounds without improving.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "LinearL2"

    _default_params = {
        "tol": 1e-6,
        "maxIter": 100,
        "aggregationDepth": 2,
        "elasticNetParam": 0.7,
        "regParam":
        [
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            5e-1,
            1,
            5,
            10,
            50,
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
        ],
        "early_stopping": 2,
    }

    def __init__(self,
                 cacher_key: str,
                 default_params: Optional[dict] = None,
                 freeze_defaults: bool = True,
                 timer: Optional[TaskTimer] = None,
                 optimization_search_space: Optional[dict] = {}):
        super().__init__(cacher_key, default_params, freeze_defaults, timer, optimization_search_space)

        self._prediction_col = f"prediction_{self._name}"
        self.task = None
        self._timer = timer
        # self._ohe = None
        self._assembler = None

        self._raw_prediction_col_name = "raw_prediction"
        self._probability_col_name = "probability"
        self._prediction_col_name = "prediction"

    def _infer_params(self,
                      train: SparkDataset,
                      fold_prediction_column: str) -> Tuple[List[Tuple[float, Estimator]], int]:
        logger.debug("Building pipeline in linear lGBFS")
        params = copy(self.params)

        if "regParam" in params:
            reg_params = params["regParam"]
            del params["regParam"]
        else:
            reg_params = [1.0]

        if "early_stopping" in params:
            es = params["early_stopping"]
            del params["early_stopping"]
        else:
            es = 100

        def build_pipeline(reg_param: int):
            instance_params = copy(params)
            instance_params["regParam"] = reg_param
            if self.task.name in ["binary", "multiclass"]:
                model = LogisticRegression(featuresCol=self._assembler.getOutputCol(),
                                           labelCol=train.target_column,
                                           probabilityCol=fold_prediction_column,
                                           rawPredictionCol=self._raw_prediction_col_name,
                                           predictionCol=self._prediction_col_name,
                                           **instance_params)
            elif self.task.name == "reg":
                model = LinearRegression(featuresCol=self._assembler.getOutputCol(),
                                         labelCol=train.target_column,
                                         predictionCol=fold_prediction_column,
                                         **instance_params)
                model = model.setSolver("l-bfgs")
            else:
                raise ValueError("Task not supported")

            return model

        estimators = [(rp, build_pipeline(rp)) for rp in reg_params]

        return estimators, es

    def fit_predict_single_fold(self,
                                fold_prediction_column: str,
                                full: SparkDataset,
                                train: SparkDataset,
                                valid: SparkDataset
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            fold_prediction_column: column name for predictions made for this fold
            full: Full dataset that include train and valid parts and a bool column that delimits records
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        logger.info(f"fit_predict single fold in LinearLBGFS. Num of features: {len(self.input_features)} ")

        if self.task is None:
            self.task = train.task

        train_sdf = train.data
        val_sdf = valid.data

        estimators, early_stopping = self._infer_params(train, fold_prediction_column)

        assert len(estimators) > 0

        es: int = 0
        best_score: float = -np.inf

        best_model: Optional[SparkMLModel] = None
        best_val_pred: Optional[SparkDataFrame] = None
        for rp, model in estimators:
            logger.debug(f"Fitting estimators with regParam {rp}")
            pipeline = Pipeline(stages=[self._assembler, model])
            ml_model = pipeline.fit(train_sdf)
            val_pred = ml_model.transform(val_sdf)
            preds_to_score = val_pred.select(
                F.col(fold_prediction_column).alias("prediction"),
                F.col(valid.target_column).alias("target")
            )
            current_score = self.score(preds_to_score)
            if current_score > best_score:
                best_score = current_score
                best_model = ml_model.stages[-1]
                best_val_pred = val_pred
                es = 0
            else:
                es += 1

            if es >= early_stopping:
                break

        logger.info("fit_predict single fold finished in LinearLBGFS")

        return best_model, best_val_pred, fold_prediction_column

    def predict_single_fold(self,
                            dataset: SparkDataset,
                            model: SparkMLModel) -> SparkDataFrame:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``SparkDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = model.transform(dataset.data)
        return pred

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        models = [el for m in self.models for el in [m, DropColumnsTransformer(
            remove_cols=[],
            optional_remove_cols=[self._prediction_col_name, self._probability_col_name, self._raw_prediction_col_name]
        )]]
        averaging_model = PipelineModel(stages=[self._assembler] + models + [avr])
        return averaging_model

    def _build_averaging_transformer(self) -> Transformer:
        avr = AveragingTransformer(
            self.task.name,
            input_cols=self._models_prediction_columns,
            output_col=self.prediction_feature,
            remove_cols=[self._assembler.getOutputCol()] + self._models_prediction_columns,
            convert_to_array_first=not (self.task.name == "reg"),
            dim_num=self.n_classes
        )
        return avr

    def fit_predict(self, train_valid_iterator: SparkBaseTrainValidIterator) -> SparkDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``numpy.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """
        logger.info("Starting LinearLGBFS")
        self.timer.start()

        self.input_roles = train_valid_iterator.input_roles
        cat_feats = [feat for feat in self.input_features if self.input_roles[feat].name == "Category"]
        # self._ohe = OneHotEncoder(inputCols=cat_feats, outputCols=[f"{f}_{self._name}_ohe" for f in cat_feats])
        # self._ohe = self._ohe.fit(train_valid_iterator.train.data)

        non_cat_feats = [feat for feat in self.input_features if self.input_roles[feat].name != "Category"]
        self._assembler = VectorAssembler(
            inputCols=non_cat_feats + cat_feats, #self._ohe.getOutputCols(),
            outputCol=f"{self._name}_vassembler_features"
        )

        result = super().fit_predict(train_valid_iterator)

        logger.info("LinearLGBFS is finished")

        return result
