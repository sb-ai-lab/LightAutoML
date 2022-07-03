import logging
from typing import Tuple, cast, List, Optional

import numpy as np
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.param import Params
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from lightautoml.dataset.roles import NumericRole, ColumnRole
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.pipelines.base import InputFeaturesAndRoles
from lightautoml.spark.utils import Cacher, SparkDataFrame
from lightautoml.spark.validation.base import SparkBaseTrainValidIterator
from lightautoml.utils.timer import TaskTimer

logger = logging.getLogger(__name__)

SparkMLModel = PipelineModel


class SparkTabularMLAlgo(MLAlgo, InputFeaturesAndRoles):
    """Machine learning algorithms that accepts numpy arrays as input."""

    _name: str = "SparkTabularMLAlgo"
    _default_validation_col_name: str = SparkBaseTrainValidIterator.TRAIN_VAL_COLUMN

    def __init__(
            self,
            cacher_key: str,
            default_params: Optional[dict] = None,
            freeze_defaults: bool = True,
            timer: Optional[TaskTimer] = None,
            optimization_search_space: Optional[dict] = {},
    ):
        super().__init__(default_params, freeze_defaults, timer, optimization_search_space)
        self._cacher_key = cacher_key
        self.n_classes: Optional[int] = None
        # names of columns that should contain predictions of individual models
        self._models_prediction_columns: Optional[List[str]] = None
        self._transformer: Optional[Transformer] = None

        # self._prediction_col = f"prediction_{self._name}"
        self._prediction_role = None

    @property
    def prediction_feature(self) -> str:
        # return self._prediction_col
        return f"prediction_{self._name}"


    @property
    def prediction_role(self) -> ColumnRole:
        return self._prediction_role

    @property
    def validation_column(self) -> str:
        return self._default_validation_col_name

    @property
    def transformer(self) -> Transformer:
        """Returns Spark MLlib Transformer.
        Represents a Transformer with fitted models."""

        assert self._transformer is not None, "Pipeline is not fitted!"

        return self._transformer

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

        logger.info(f"Input columns for MLALgo: {sorted(train_valid_iterator.input_features)}")
        logger.info(f"Train size for MLAlgo: {train_valid_iterator.train.data.count()}")

        # log_data(f"spark_fit_predict_{type(self).__name__}", {"train": train_valid_iterator.train.to_pandas()})

        assert self.is_fitted is False, "Algo is already fitted"
        # init params on input if no params was set before
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)

        iterator_len = len(train_valid_iterator)
        if iterator_len > 1:
            logger.info("Start fitting \x1b[1m{}\x1b[0m ...".format(self._name))
            logger.debug(f"Training params: {self.params}")

        # save features names
        self._features = train_valid_iterator.features
        # get metric and loss if None
        self.task = train_valid_iterator.train.task

        valid_ds = cast(SparkDataset, train_valid_iterator.get_validation_data())

        # spark
        prob = train_valid_iterator.train.task.name in ["binary", "multiclass"]
        outp_dim = 1
        if self.task.name == "multiclass":
            outp_dim = valid_ds.data.select(F.max(valid_ds.target_column).alias("max")).first()
            outp_dim = outp_dim["max"] + 1
            self._prediction_role = NumericVectorOrArrayRole(outp_dim,
                                                             f"{self.prediction_feature}" + "_{}",
                                                             np.float32,
                                                             force_input=True,
                                                             prob=True)
        elif self.task.name == "binary":
            outp_dim = 2
            self._prediction_role = NumericVectorOrArrayRole(outp_dim,
                                                             f"{self.prediction_feature}" + "_{}",
                                                             np.float32,
                                                             force_input=True,
                                                             prob=True)
        else:
            self._prediction_role = NumericRole(np.float32, force_input=True, prob=False)

        self.n_classes = outp_dim

        preds_dfs: List[SparkDataFrame] = []

        pred_col_prefix = self._predict_feature_name()

        self._models_prediction_columns = []
        for n, (full, train, valid) in enumerate(train_valid_iterator):
            if iterator_len > 1:
                logger.info2(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m =====".format(n, self._name)
                )
            self.timer.set_control_point()

            model_prediction_col = f"{pred_col_prefix}_{n}"
            model, val_pred, _ = self.fit_predict_single_fold(model_prediction_col, full, train, valid)

            self._models_prediction_columns.append(model_prediction_col)
            self.models.append(model)
            preds_dfs.append(val_pred)

            self.timer.write_run_info()

            if (n + 1) != len(train_valid_iterator):
                # split into separate cases because timeout checking affects parent pipeline timer
                if self.timer.time_limit_exceeded():
                    logger.info("Time limit exceeded after calculating fold {0}\n".format(n))
                    break

        # combining results for different folds
        # 1. folds - union
        # 2. dummy - nothing
        # 3. holdout - nothing
        # 4. custom - union + groupby
        # neutral_element = (
        #     array_to_vector(F.array(*[F.lit(float('nan')) for _ in range(self.n_classes)]))
        #     if self.task.name in ["binary", "multiclass"]
        #     else F.lit(float('nan'))
        # )

        neutral_element = None

        preds_dfs = [
            df.select(
                '*',
                *[F.lit(neutral_element).alias(c) for c in self._models_prediction_columns if c not in df.columns]
            )
            for df in preds_dfs
        ]
        full_preds_df = train_valid_iterator.combine_val_preds(preds_dfs, include_train=False)
        full_preds_df = self._build_averaging_transformer().transform(full_preds_df)

        cacher = Cacher(key=self._cacher_key)
        cacher.fit(full_preds_df)
        full_preds_df = cacher.dataset

        # create Spark MLlib Transformer and save to property var
        self._transformer = self._build_transformer()

        pred_ds = self._set_prediction(valid_ds.empty(), full_preds_df)

        if iterator_len > 1:
            single_pred_ds = self._make_single_prediction_dataset(pred_ds)
            logger.info(
                f"Fitting \x1b[1m{self._name}\x1b[0m finished. score = \x1b[1m{self.score(single_pred_ds)}\x1b[0m")

        if iterator_len > 1 or "Tuned" not in self._name:
            logger.info("\x1b[1m{}\x1b[0m fitting and predicting completed".format(self._name))

        return pred_ds

    def fit_predict_single_fold(self,
                                fold_prediction_column: str,
                                full: SparkDataset,
                                train: SparkDataset,
                                valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            fold_prediction_column: column name for predictions made for this fold
            full: Full dataset that include train and valid parts and a bool column that delimits records
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        raise NotImplementedError

    def predict_single_fold(self, model: SparkMLModel, dataset: SparkDataset) -> SparkDataFrame:
        raise NotImplementedError("Not supported for Spark. Use transformer property instead ")
        pass

    def predict(self, dataset: SparkDataset) -> SparkDataset:
        sdf = self.transformer.transform(dataset.data)

        ds = dataset.empty()
        ds.set_data(sdf, [self.prediction_feature], self.prediction_role)

        return ds

    @staticmethod
    def _get_predict_column(model: SparkMLModel) -> str:
        try:
            return model.getPredictionCol()
        except AttributeError:
            if isinstance(model, PipelineModel):
                return model.stages[-1].getPredictionCol()

            raise TypeError("Unknown model type! Unable ro retrieve prediction column")

    def _predict_feature_name(self):
        return f"{self._name}_prediction"

    def _set_prediction(self, dataset: SparkDataset,  preds: SparkDataFrame) -> SparkDataset:
        """Insert predictions to dataset with. Inplace transformation.

        Args:
            dataset: Dataset to transform.
            preds: A spark dataframe  with predicted values.

        Returns:
            Transformed dataset.

        """

        prob = self.task.name in ["binary", "multiclass"]

        if self.task.name in ["binary", "multiclass"]:
            role = NumericVectorOrArrayRole(size=self.n_classes,
                                            element_col_name_template=self._predict_feature_name() + "_{}",
                                            dtype=np.float32,
                                            force_input=True,
                                            prob=prob)
        else:
            role = NumericRole(dtype=np.float32, force_input=True, prob=prob)

        output: SparkDataset = dataset.empty()
        output.set_data(preds, [self._predict_feature_name()], role)

        return output

    def _build_transformer(self) -> Transformer:
        raise NotImplementedError()

    def _build_averaging_transformer(self) -> Transformer:
        raise NotImplementedError()

    def _make_single_prediction_dataset(self, dataset: SparkDataset) -> SparkDataset:
        preds = dataset.data.select(SparkDataset.ID_COLUMN, dataset.target_column, self.prediction_feature)
        roles = {self.prediction_feature: dataset.roles[self.prediction_feature]}

        output: SparkDataset = dataset.empty()
        output.set_data(preds, preds.columns, roles)

        return output


class AveragingTransformer(Transformer, HasInputCols, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):
    """
    Transformer that gets one or more columns and produce column with average values.
    """
    taskName = Param(Params._dummy(), "taskName", "task name")
    removeCols = Param(Params._dummy(), "removeCols", "cols to remove")
    convertToArrayFirst = Param(Params._dummy(), "convertToArrayFirst", "convert to array first")
    weights = Param(Params._dummy(), "weights", "weights")
    dimNum = Param(Params._dummy(), "dimNum", "dim num")

    def __init__(self,
                 task_name: str = None,
                 input_cols: List[str] = [],
                 output_col: str = "averaged_values",
                 remove_cols: Optional[List[str]] = None,
                 convert_to_array_first: bool = False,
                 weights: Optional[List[int]] = None,
                 dim_num: int = 1):
        """
        Args:
            task_name (str, optional): Task name: "binary", "multiclass" or "reg".
            input_cols (List[str], optional): List of input columns.
            output_col (str, optional): Output column name. Defaults to "averaged_values".
            remove_cols (Optional[List[str]], optional): Columns need to remove. Defaults to None.
            convert_to_array_first (bool, optional): If `True` then will be convert input vectors to arrays. Defaults to False.
            weights (Optional[List[int]], optional): List of weights to scaling output values. Defaults to None.
            dim_num (int, optional): Dimension of input columns. Defaults to 1.
        """
        super().__init__()
        self.set(self.taskName, task_name)
        self.set(self.inputCols, input_cols)
        self.set(self.outputCol, output_col)
        if not remove_cols:
            remove_cols = []
        self.set(self.removeCols, remove_cols)
        self.set(self.convertToArrayFirst, convert_to_array_first)
        if weights is None:
            weights = [1.0 for _ in input_cols]

        assert len(input_cols) == len(weights)

        self.set(self.weights, weights)
        self.set(self.dimNum, dim_num)

    def getTaskName(self) -> str:
        return self.getOrDefault(self.taskName)

    def getRemoveCols(self) -> List[str]:
        return self.getOrDefault(self.removeCols)

    def getConvertToArrayFirst(self) -> bool:
        return self.getOrDefault(self.convertToArrayFirst)

    def getWeights(self) -> List[int]:
        return self.getOrDefault(self.weights)

    def getDimNum(self) -> int:
        return self.getOrDefault(self.dimNum)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.info(f"In transformer {type(self)}. Columns: {sorted(dataset.columns)}")

        pred_cols = self.getInputCols()
        dim_size = self.getInputCols()
        weights = {c: w for w, c in zip(self.getWeights(), pred_cols)}
        non_null_count_col = (F.lit(len(pred_cols)) - sum(F.isnull(c).astype(IntegerType()) for c in pred_cols))

        if self.getTaskName() in ["binary", "multiclass"]:
            def convert_column(c):
                return vector_to_array(c).alias(c) if self.getConvertToArrayFirst() else F.col(c)

            normalized_cols = [
                F.when(F.isnull(c), F.array(*[F.lit(0.0) for _ in range(self.getDimNum())]))
                    .otherwise(convert_column(c)).alias(c)
                for c in pred_cols
            ]
            arr_fields_summ = F.transform(F.arrays_zip(*normalized_cols), lambda x: F.aggregate(
                F.array(*[x[c] * F.lit(weights[c]) for c in pred_cols]),
                F.lit(0.0),
                lambda acc, x: acc + x
            ) / non_null_count_col)

            out_col = array_to_vector(arr_fields_summ) if self.getConvertToArrayFirst() else arr_fields_summ
        else:
            scalar_fields_summ = F.aggregate(
                F.array(*[F.col(c) * F.lit(weights[c]) for c in pred_cols]),
                F.lit(0.0),
                lambda acc, x: acc + F.when(F.isnull(x), F.lit(0.0)).otherwise(x)
            ) / non_null_count_col

            out_col = scalar_fields_summ

        cols_to_remove = set(self.getRemoveCols())
        cols_to_select = [c for c in dataset.columns if c not in cols_to_remove]
        out_df = dataset.select(*cols_to_select, out_col.alias(self.getOutputCol()))

        logger.info(f"In the end of transformer {type(self)}. Columns: {sorted(dataset.columns)}")

        return out_df
