# https://github.com/sllynn/spark-xgboost/blob/master/examples/spark-xgboost_adultdataset.ipynb
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
from pyspark.sql import SparkSession
from sparkxgb import XGBoostClassifier, XGBoostRegressor
from pprint import PrettyPrinter

from pyspark.sql.types import StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = (
    SparkSession
    .builder
    .master('local[2]')
    .config("spark.jars.packages", "ml.dmlc:xgboost4j_2.12:1.5.2,ml.dmlc:xgboost4j-spark_2.12:1.5.2")
    .getOrCreate()
)

pp = PrettyPrinter()


col_names = [
  "age", "workclass", "fnlwgt",
  "education", "education-num",
  "marital-status", "occupation",
  "relationship", "race", "sex",
  "capital-gain", "capital-loss",
  "hours-per-week", "native-country",
  "label"
]

train_sdf, test_sdf = (
  spark.read.csv(
    path="file:///opt/spark_data/adult.data",
    inferSchema=True
  )
  .toDF(*col_names)
  .repartition(200)
  .randomSplit([0.8, 0.2])
)

string_columns = [fld.name for fld in train_sdf.schema.fields if isinstance(fld.dataType, StringType)]
string_col_replacements = [fld + "_ix" for fld in string_columns]
string_column_map=list(zip(string_columns, string_col_replacements))
target = string_col_replacements[-1]
predictors = [fld.name for fld in train_sdf.schema.fields if not isinstance(fld.dataType, StringType)] + string_col_replacements[:-1]
pp.pprint(
  dict(
    string_column_map=string_column_map,
    target_variable=target,
    predictor_variables=predictors
  )
)


si = [StringIndexer(inputCol=fld[0], outputCol=fld[1]) for fld in string_column_map]
va = VectorAssembler(inputCols=predictors, outputCol="features")
pipeline = Pipeline(stages=[*si, va])
fitted_pipeline = pipeline.fit(train_sdf.union(test_sdf))
train_sdf_prepared = fitted_pipeline.transform(train_sdf)
test_sdf_prepared = fitted_pipeline.transform(test_sdf)


xgbParams = dict(
  eta=0.1,
  maxDepth=2,
  missing=0.0,
  objective="binary:logistic",
  numRound=5,
  numWorkers=2
)

xgb = (
  XGBoostClassifier(**xgbParams)
  .setFeaturesCol("features")
  .setLabelCol("label_ix")
)

bce = BinaryClassificationEvaluator(
  rawPredictionCol="rawPrediction",
  labelCol="label_ix"
)

model = xgb.fit(train_sdf_prepared)
score = bce.evaluate(model.transform(test_sdf_prepared))

print(f"Test score: {score}")

spark.stop()