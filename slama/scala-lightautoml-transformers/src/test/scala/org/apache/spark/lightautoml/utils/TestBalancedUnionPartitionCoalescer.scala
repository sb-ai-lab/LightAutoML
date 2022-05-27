package org.apache.spark.lightautoml.utils

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Row, SparkSession}

import scala.util.Random

/*
*  To run this example:
*   - set env variable: SPARK_SCALA_VERSION=2.12
*   - create 'assembly/target/scala-2.12/jars' directory in the root of the project (scala-lightautoml-transformers)
*   - copy spark jars to 'assembly/target/scala-2.12/jars'. These jars can be taken from spark/pyspark distributions
*     for example: cp -r $HOME/.cache/pypoetry/virtualenvs/lightautoml-749ciRtl-py3.9/lib/python3.9/site-packages/pyspark/jars/ assembly/target/scala-2.12/jars
*   - ensure that spark-lightautoml_2.12-0.1.jar has been built and accessible for spark (run: sbt package)
* */
object TestBalancedUnionPartitionCoalescer extends App {
  val num_workers = 3
  val num_cores = 2
  val folds_count = 5

  val spark = SparkSession
          .builder()
          .master(s"local-cluster[${num_workers}, ${num_cores}, 1024]")
          .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.jar")
          .getOrCreate()

  import spark.sqlContext.implicits._

  val df = spark
          .sparkContext.parallelize((0 until 5000)
          .map(x => (x, Random.nextInt(folds_count)))).toDF("data", "fold")
          .repartition(num_workers * num_cores * 2)
          .cache()
  df.write.mode("overwrite").format("noop").save()

  val dfs = (0 until 5).map(x => df.where(col("fold").equalTo(x)))
  val full_df = dfs.reduce((acc, sdf) => acc.unionByName(sdf))

  val unionPartitionsCoalescerTransformer = new BalancedUnionPartitionsCoalescerTransformer(uid = "some uid")
  var coalesced_df = unionPartitionsCoalescerTransformer.transform(full_df)

  coalesced_df = coalesced_df.cache()
  coalesced_df.write.mode("overwrite").format("noop").save()

  coalesced_df.count()

  val result = coalesced_df.rdd.collectPartitions()

  // checking:
  // 1. Number of partitions should be the same as the number of partitions in the initial dataframe
  val sameNumOfPartitions = df.rdd.getNumPartitions == coalesced_df.rdd.getNumPartitions
  assert(sameNumOfPartitions)

  // 2. all partitions should have approximately the same number of records
  val parts_sizes = result.map(_.length)
  val min_size = parts_sizes.min
  val max_size = parts_sizes.max
  assert((max_size - min_size).toFloat / min_size <= 0.02)

  def parts2folds(res: Array[Array[Row]]): Map[Int, Map[Int, Int]] = {
    res
      .zipWithIndex
      .flatMap(x => x._1.map(y => (x._2, y.getInt(1))))
      .sortBy(_._1)
      .groupBy(_._1)
      .map(x => (x._1, x._2.map(_._2)))
      .map(x => (x._1, x._2.sortBy(x => x).groupBy(x => x).map(x => (x._1, x._2.length))))
  }

  // part_id, (fold, record_count)
  val partsWithFolds = parts2folds(result)

  // 3. All folds should be presented in all partitions of the resulting dataframe
  val allFoldsInAllPartitions = partsWithFolds.forall(_._2.size == folds_count)
  assert(allFoldsInAllPartitions)

  // 4. The resulting dataframe should contain the same number of records of a certain fold
  // as it appears in parent datasets
  for (f_df <- dfs) {
    val f_parts = f_df.rdd.collectPartitions()
    val single_fold_parts = parts2folds(f_parts)

    val allPartsContainSingleFold = single_fold_parts.forall(_._2.size == 1)
    assert(allPartsContainSingleFold)

    val countOfRecordsWithCertainFoldsShouldStayTheSame = partsWithFolds.forall{ x =>
      val (fold_num, count) = single_fold_parts(x._1).head
      val count_after_union = x._2(fold_num)
      count == count_after_union
    }

    assert(countOfRecordsWithCertainFoldsShouldStayTheSame)
  }

  spark.stop()
}
