package org.apache.spark.lightautoml.utils

import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD, UnionPartition}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.util.Random


class BalancedUnionPartitionCoalescer extends PartitionCoalescer with Serializable {
  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val up_arr = parent.partitions.map(_.asInstanceOf[UnionPartition[_]])
    val parent2parts = up_arr
            .map(x => (x.parentRddIndex, x))
            .groupBy(_._1)
            .map(x => (x._1, x._2.map(_._2).sortBy(_.parentPartition.index)))

    val unique_sizes = parent2parts.map(_._2.length).toSet

    assert(unique_sizes.size == 1)

    val partsNum = unique_sizes.head

//    assert(maxPartitions <= partsNum)

    val pgs = (0 until partsNum).map(i => {
      val pg = new PartitionGroup()
      parent2parts.values.foreach(x => pg.partitions += x(i))
      pg
    })

    pgs.toArray
  }
}

