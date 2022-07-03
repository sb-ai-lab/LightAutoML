import logging
import multiprocessing
import warnings
from copy import copy
from typing import Dict, Optional, Tuple, Union, cast

import pandas as pd
import pyspark.sql.functions as F
from pandas import Series
from pyspark.ml import Transformer, PipelineModel
from pyspark.ml.util import MLWritable, MLReadable, MLWriter
from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor, LightGBMRegressionModel, LightGBMClassificationModel
from synapse.ml.onnx import ONNXModel
import lightgbm as lgb
from lightgbm import Booster, LGBMClassifier


from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.utils import SparkDataFrame
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, AveragingTransformer
from lightautoml.spark.mlwriters import LightGBMModelWrapperMLReader, LightGBMModelWrapperMLWriter, ONNXModelWrapperMLReader, ONNXModelWrapperMLWriter
from lightautoml.spark.transformers.base import DropColumnsTransformer, PredictionColsTransformer, ProbabilityColsTransformer
from lightautoml.spark.validation.base import SparkBaseTrainValidIterator
from lightautoml.utils.timer import TaskTimer
from lightautoml.validation.base import TrainValidIterator

logger = logging.getLogger(__name__)


class LightGBMModelWrapper(Transformer, MLWritable, MLReadable):
    """Simple wrapper for `synapse.ml.lightgbm.[LightGBMRegressionModel|LightGBMClassificationModel]` to fix issue with loading model from saved composite pipeline.
    
    For more details see: https://github.com/microsoft/SynapseML/issues/614.
    """

    def __init__(self, model: Union[LightGBMRegressionModel, LightGBMClassificationModel] = None) -> None:
        super().__init__()
        self.model = model

    def write(self) -> MLWriter:
        return LightGBMModelWrapperMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return LightGBMModelWrapperMLReader()

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        return self.model.transform(dataset)


class ONNXModelWrapper(Transformer, MLWritable, MLReadable):
    """Simple wrapper for `ONNXModel` to fix issue with loading model from saved composite pipeline.
    
    For more details see: https://github.com/microsoft/SynapseML/issues/614.
    """

    def __init__(self, model: ONNXModel = None) -> None:
        super().__init__()
        self.model = model

    def write(self) -> MLWriter:
        return ONNXModelWrapperMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return ONNXModelWrapperMLReader()

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        return self.model.transform(dataset)
 

class SparkBoostLGBM(SparkTabularMLAlgo, ImportanceEstimator):
    """Gradient boosting on decision trees from LightGBM library.

    default_params: All available parameters listed in synapse.ml documentation:

        - https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMClassifier
        - https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMRegressor

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "LightGBM"

    _default_params = {
        "learningRate": 0.05,
        "numLeaves": 128,
        "featureFraction": 0.7,
        "baggingFraction": 0.7,
        "baggingFreq": 1,
        "maxDepth": -1,
        "minGainToSplit": 0.0,
        "maxBin": 255,
        "minDataInLeaf": 5,
        # e.g. num trees
        "numIterations": 3000,
        "earlyStoppingRound": 50,
        # for regression
        "alpha": 1.0,
        "lambdaL1": 0.0,
        "lambdaL2": 0.0,
    }

    # mapping between metric name defined via SparkTask
    # and metric names supported by LightGBM
    _metric2lgbm = {
        "binary": {
            "auc": "auc",
            "aupr": "areaUnderPR"
        },
        "reg": {
            "r2": "rmse",
            "mse": "mse",
            "mae": "mae",
        },
        "multiclass": {
            "crossentropy": "cross_entropy"
        }
    }

    def __init__(self,
                 cacher_key: str,
                 default_params: Optional[dict] = None,
                 freeze_defaults: bool = True,
                 timer: Optional[TaskTimer] = None,
                 optimization_search_space: Optional[dict] = {},
                 use_single_dataset_mode: bool = True,
                 max_validation_size: int = 10_000,
                 chunk_size: int = 4_000_000,
                 convert_to_onnx: bool = False,
                 mini_batch_size: int = 5000,
                 seed: int = 42):
        SparkTabularMLAlgo.__init__(self, cacher_key, default_params, freeze_defaults, timer, optimization_search_space)
        self._probability_col_name = "probability"
        self._prediction_col_name = "prediction"
        self._raw_prediction_col_name = "raw_prediction"
        self._assembler = None
        self._drop_cols_transformer = None
        self._use_single_dataset_mode = use_single_dataset_mode
        self._max_validation_size = max_validation_size
        self._seed = seed
        self._models_feature_impotances = []
        self._chunk_size = chunk_size
        self._convert_to_onnx = convert_to_onnx
        self._mini_batch_size = mini_batch_size

    def _infer_params(self) -> Tuple[dict, int]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        assert self.task is not None

        task = self.task.name

        params = copy(self.params)

        if "isUnbalance" in params:
            params["isUnbalance"] = True if params["isUnbalance"] == 1 else False

        verbose_eval = 1

        if task == "reg":
            params["objective"] = "regression"
            params["metric"] = self._metric2lgbm[task].get(self.task.metric_name, None)
        elif task == "binary":
            params["objective"] = "binary"
            params["metric"] = self._metric2lgbm[task].get(self.task.metric_name, None)
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["metric"] = "multiclass"
        else:
            raise ValueError(f"Unsupported task type: {task}")

        if task != "reg":
            if "alpha" in params:
                del params["alpha"]
            if "lambdaL1" in params:
                del params["lambdaL1"]
            if "lambdaL2" in params:
                del params["lambdaL2"]

        params = {**params}

        return params, verbose_eval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        self.task = train_valid_iterator.train.task

        sds = cast(SparkDataset, train_valid_iterator.train)
        rows_num = sds.data.count()
        task = train_valid_iterator.train.task.name

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if task == "reg":
            suggested_params = {
                "learningRate": 0.05,
                "numLeaves": 32,
                "featureFraction": 0.9,
                "baggingFraction": 0.9,
            }

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200

        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200

        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 2000
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            suggested_params["numLeaves"] = 128 if task == "reg" else 244
        elif rows_num > 100000:
            suggested_params["numLeaves"] = 64 if task == "reg" else 128
        elif rows_num > 50000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 64
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 32
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.0
        elif rows_num > 10000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 64
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.2
        elif rows_num > 5000:
            suggested_params["numLeaves"] = 24 if task == "reg" else 32
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.5
        else:
            suggested_params["numLeaves"] = 16 if task == "reg" else 16
            suggested_params["alpha"] = 1 if task == "reg" else 1

        suggested_params["learningRate"] = init_lr
        suggested_params["numIterations"] = ntrees
        suggested_params["earlyStoppingRound"] = es

        if task != "reg":
            if "alpha" in suggested_params:
                del suggested_params["alpha"]

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Train on train dataset and predict on holdout dataset.

        Args:
            fold_prediction_column: column name for predictions made for this fold
            full: Full dataset that include train and valid parts and a bool column that delimits records
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        assert self.task is not None

        optimization_search_space = dict()

        optimization_search_space["featureFraction"] = SearchSpace(
            Distribution.UNIFORM,
            low=0.5,
            high=1.0,
        )

        optimization_search_space["numLeaves"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=4,
            high=255,
        )

        if self.task.name == "binary" or self.task.name == "multiclass":
            optimization_search_space["isUnbalance"] = SearchSpace(
                Distribution.DISCRETEUNIFORM,
                low=0,
                high=1,
                q=1
            )

        if estimated_n_trials > 30:
            optimization_search_space["baggingFraction"] = SearchSpace(
                Distribution.UNIFORM,
                low=0.5,
                high=1.0,
            )

            optimization_search_space["minSumHessianInLeaf"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-3,
                high=10.0,
            )

        if estimated_n_trials > 100:
            if self.task.name == "reg":
                optimization_search_space["alpha"] = SearchSpace(
                    Distribution.LOGUNIFORM,
                    low=1e-8,
                    high=10.0,
                )

            optimization_search_space["lambdaL1"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )

        return optimization_search_space

    def predict_single_fold(self,
                            dataset: SparkDataset,
                            model: Union[LightGBMRegressor, LightGBMClassifier]) -> SparkDataFrame:

        temp_sdf = self._assembler.transform(dataset.data)

        pred = model.transform(temp_sdf)

        return pred

    def fit_predict_single_fold(self,
                                fold_prediction_column: str,
                                full: SparkDataset,
                                train: SparkDataset,
                                valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        assert self.validation_column in full.data.columns, 'Train should contain validation column'

        if self.task is None:
            self.task = full.task

        (
            params,
            verbose_eval
        ) = self._infer_params()

        logger.info(f"Input cols for the vector assembler: {full.features}")
        logger.info(f"Running lgb with the following params: {params}")

        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=self.input_features,
                outputCol=f"{self._name}_vassembler_features",
                handleInvalid="keep"
            )

        LGBMBooster = LightGBMRegressor if full.task.name == "reg" else LightGBMClassifier

        if full.task.name in ['binary', 'multiclass']:
            params['rawPredictionCol'] = self._raw_prediction_col_name
            params['probabilityCol'] = fold_prediction_column
            params['predictionCol'] = self._prediction_col_name
            params['isUnbalance'] = True
        else:
            params['predictionCol'] = fold_prediction_column

        master_addr = train.spark_session.conf.get('spark.master')
        if master_addr.startswith('local'):
            cores_str = master_addr[len("local["):-1]
            cores = int(cores_str) if cores_str != "*" else multiprocessing.cpu_count()
            params["numThreads"] = max(cores - 1, 1)
        else:
            params["numThreads"] = max(int(train.spark_session.conf.get("spark.executor.cores", "1")) - 1, 1)

        train_data = full.data
        valid_size = train_data.where(F.col(self.validation_column) == 1).count()
        max_val_size = self._max_validation_size
        if valid_size > max_val_size:
            warnings.warn(f"Maximum validation size for SparkBoostLGBM is exceeded: {valid_size} > {max_val_size}. "
                          f"Reducing validation size down to maximum.", category=RuntimeWarning)
            rest_cols = list(train_data.columns)
            rest_cols.remove(self.validation_column)

            replace_col = F.when(F.rand(self._seed) < max_val_size / valid_size, F.lit(True)).otherwise(F.lit(False))
            val_filter_cond = F.when(F.col(self.validation_column) == 1, replace_col).otherwise(F.lit(True))

            train_data = train_data.where(val_filter_cond)

        valid_data = valid.data

        lgbm = LGBMBooster(
            **params,
            featuresCol=self._assembler.getOutputCol(),
            labelCol=full.target_column,
            validationIndicatorCol=self.validation_column,
            verbosity=verbose_eval,
            useSingleDatasetMode=self._use_single_dataset_mode,
            isProvideTrainingMetric=True,
            chunkSize=self._chunk_size
        )

        logger.info(f"Use single dataset mode: {lgbm.getUseSingleDatasetMode()}. NumThreads: {lgbm.getNumThreads()}")

        if full.task.name == "reg":
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        ml_model = lgbm.fit(self._assembler.transform(train_data))

        val_pred = ml_model.transform(self._assembler.transform(valid_data))
        val_pred = DropColumnsTransformer(
            remove_cols=[],
            optional_remove_cols=[self._prediction_col_name, self._probability_col_name, self._raw_prediction_col_name]
        ).transform(val_pred)

        self._models_feature_impotances.append(ml_model.getFeatureImportances(importance_type='gain'))

        if self._convert_to_onnx:
            logger.info("Model convert is started")
            booster_model_str = ml_model.getLightGBMBooster().modelStr().get()
            booster = lgb.Booster(model_str=booster_model_str)
            model_payload_ml = self._convertModel(booster, len(self.input_features))

            onnx_ml = ONNXModel().setModelPayload(model_payload_ml)

            if full.task.name == "reg":
                onnx_ml = (
                    onnx_ml
                        .setDeviceType("CPU")
                        .setFeedDict({"input": f"{self._name}_vassembler_features"})
                        .setFetchDict({ml_model.getPredictionCol(): "variable"})
                        .setMiniBatchSize(self._mini_batch_size)
                )
            else:
                onnx_ml = (
                    onnx_ml
                        .setDeviceType("CPU")
                        .setFeedDict({"input": f"{self._name}_vassembler_features"})
                        .setFetchDict({ml_model.getProbabilityCol(): "probabilities", ml_model.getPredictionCol(): "label"})
                        .setMiniBatchSize(self._mini_batch_size)
                )

            ml_model = onnx_ml
            logger.info("Model convert is ended")

        return ml_model, val_pred, fold_prediction_column

    def fit(self, train_valid: SparkBaseTrainValidIterator):
        logger.info("Starting LGBM fit")
        self.fit_predict(train_valid)
        logger.info("Finished LGBM fit")

    def get_features_score(self) -> Series:
        imp = 0
        for model_feature_impotances in self._models_feature_impotances:
            imp = imp + pd.Series(model_feature_impotances)

        imp = imp / len(self._models_feature_impotances)

        result = Series(list(imp), index=self.features).sort_values(ascending=False)
        return result

    @staticmethod
    def _convertModel(lgbm_model: Booster, input_size: int) -> bytes:
        from onnxmltools.convert import convert_lightgbm
        from onnxconverter_common.data_types import FloatTensorType
        initial_types = [("input", FloatTensorType([-1, input_size]))]
        onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset=9)
        return onnx_model.SerializeToString()

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        if self._convert_to_onnx:
            wrapped_models = [ONNXModelWrapper(m) for m in self.models]
        else:
            wrapped_models = [LightGBMModelWrapper(m) for m in self.models]
        models = [el for m in wrapped_models for el in [m, DropColumnsTransformer(
                remove_cols=[],
                optional_remove_cols=[self._prediction_col_name,
                                    self._probability_col_name,
                                    self._raw_prediction_col_name]
            )
        ]]
        if self._convert_to_onnx:
            if self.task.name in ['binary', 'multiclass']:
                models.append(ProbabilityColsTransformer(probability_сols=self._models_prediction_columns, num_classes=self.n_classes))
            else:
                models.append(PredictionColsTransformer(prediction_сols=self._models_prediction_columns))
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
        logger.info("Starting LGBM fit")
        self.timer.start()

        self.input_roles = train_valid_iterator.input_roles
        
        res = super().fit_predict(train_valid_iterator)

        logger.info("Finished LGBM fit")
        return res
