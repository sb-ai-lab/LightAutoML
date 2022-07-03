package org.apache.spark.lightautoml.utils

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.StructType

/**
 *  Extremely specific transformer to make balanced partitions, equally filled with records belonging
 *  to different folds.
 *  !!! Can be used only with dataset produced by union of multiple datasets
 *  each containing records belonging to a single fold !!!
 * @param uid - unique id of the transformer
 */
class BalancedUnionPartitionsCoalescerTransformer(override val uid: String) extends Transformer {
  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.active
    val ds = dataset.asInstanceOf[Dataset[Row]]

    // real numPartitions is identified from the incoming dataset
    val coalesced_rdd = ds.rdd.coalesce(
      numPartitions = 100,
      shuffle = false,
      partitionCoalescer = Some(new BalancedUnionPartitionCoalescer)
    )

    val coalesced_df = spark.createDataFrame(coalesced_rdd, schema = dataset.schema)

    coalesced_df
  }

  override def copy(extra: ParamMap): Transformer = new BalancedUnionPartitionsCoalescerTransformer(uid)

  override def transformSchema(schema: StructType): StructType = schema.copy()
}
