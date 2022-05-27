package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.feature.lightautoml.{LAMLStringIndexer, LAMLStringIndexerModel}


object TestLAMLStringIndexer extends App {

  val file = "resources/data.json"
  val testFile = "resources/test_data.json"
  val spark = SparkSession
          .builder()
          .appName("test")
          .master("local[1]")
          .getOrCreate()

  //import spark.sqlContext.implicits._

  val df = spark.read.json(file).cache()
  val testDf = spark.read.json(testFile).cache()
  val cnt = df.count()

  println("-- Source --")
  df.show(100)

  val startTime = System.currentTimeMillis()

  val lamaIndexer = new LAMLStringIndexer()
          .setMinFreq(Array(1))
          .setFreqLabel(true)
          .setDefaultValue(1.0F)
          .setInputCols(Array("value"))
          .setOutputCols(Array("index"))
          .setHandleInvalid("keep")

  //val _lamaModelTestNoRuntimeError = new LAMLStringIndexerModel(labelsArray = Array(Array(("a", 1), ("b", 2))))

  println(lamaIndexer.uid)

  val lamaModel = lamaIndexer.fit(df)
  val lamaTestIndexed = lamaModel.transform(testDf)

  println("-- Lama Indexed --")
  lamaTestIndexed.show(100)

  val endTime = System.currentTimeMillis()

  println(s"Duration = ${(endTime - startTime) / 1000D} seconds")
  println(s"Size: ${cnt}")

//  println(s"[${indexer.uid} - ${model.uid}] // [${lamaIndexer.uid} - ${lamaModel.uid}]")

  lamaModel.write.overwrite().save("/tmp/LAMLStringIndexerModel")
  val pipelineModel = LAMLStringIndexerModel.load("/tmp/LAMLStringIndexerModel")
  pipelineModel.transform(testDf)

//  while (args(0).toBoolean) {
//    Thread.sleep(1000)
//  }
}
