import logging.config
import logging.config

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel

from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_time
from lightautoml.spark.validation.iterators import SparkFoldsIterator

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 5
    dataset_name = "lama_test_dataset"
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }
    cacher_key = "main_cache"

    with log_exec_time():
        train_df, test_df = prepare_test_and_train(spark, path, seed)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(train_df, roles=roles)

        iterator = SparkFoldsIterator(sdataset, n_folds=cv)

        spark_ml_algo = SparkBoostLGBM(cacher_key=cacher_key, freeze_defaults=False)
        spark_features_pipeline = SparkLGBAdvancedPipeline(cacher_key=cacher_key, **ml_alg_kwargs)
        spark_selector = ImportanceCutoffSelector(
            cutoff=0.0,
            feature_pipeline=SparkLGBSimpleFeatures(cacher_key='preselector'),
            ml_algo=SparkBoostLGBM(cacher_key='preselector', freeze_defaults=False),
            imp_estimator=ModelBasedImportanceEstimator()
        )

        ml_pipe = SparkMLPipeline(
            cacher_key=cacher_key,
            ml_algos=[spark_ml_algo],
            pre_selection=spark_selector,
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        oof_preds_ds = ml_pipe.fit_predict(iterator)
        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

        # 1. first way (LAMA API)
        test_sds = sreader.read(test_df, add_array_attrs=True)
        test_preds_ds = ml_pipe.predict(test_sds)
        test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"Test score (#1 way): {test_score}")

        # 2. second way (Spark ML API)
        transformer = PipelineModel(stages=[sreader.make_transformer(add_array_attrs=True), ml_pipe.transformer])
        test_pred_df = transformer.transform(test_df)
        test_pred_df = test_pred_df.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias('prediction')
        )
        test_score = score(test_pred_df)
        logger.info(f"Test score (#2 way): {test_score}")

        transformer.write().overwrite().save("/tmp/reader_and_spark_ml_pipe_lgb")

        # 3. third way (via loaded Spark ML Pipeline)
        pipeline_model = PipelineModel.load("/tmp/reader_and_spark_ml_pipe_lgb")
        test_pred_df = pipeline_model.transform(test_df)
        test_pred_df = test_pred_df.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias('prediction')
        )
        test_score = score(test_pred_df)
        logger.info(f"Test score (#3 way): {test_score}")

    logger.info("Finished")

    spark.stop()
