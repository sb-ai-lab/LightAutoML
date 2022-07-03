package org.apache.spark.ml.feature.lightautoml

import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkContext, SparkException}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerAggregator, StringIndexerBase, StringIndexerModel}
import org.apache.spark.annotation.Since
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.expressions.{GenericRowWithSchema, If, Literal}
import org.apache.spark.sql.functions.{collect_set, udf, lit}
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{Column, DataFrame, Dataset, Encoder, Encoders, SparkSession}
import org.apache.spark.util.ThreadUtils
import org.apache.spark.util.VersionUtils.majorMinorVersion
import org.apache.spark.util.collection.OpenHashMap

//import java.util

private[lightautoml] trait LAMLStringIndexerBase extends StringIndexerBase {
  // Overridden from StringIndexerBase due to we are able to handle Boolean columns.
  private def validateAndTransformField(schema: StructType,
                                        inputColName: String,
                                        outputColName: String): StructField = {
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == StringType || inputDataType.isInstanceOf[NumericType] || inputDataType == BooleanType,
      s"The input column $inputColName must be string type, boolean type or numeric type, " +
              s"but got $inputDataType.")
    require(schema.fields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    NominalAttribute.defaultAttr.withName(outputColName).toStructField()
  }

  override protected def validateAndTransformSchema(schema: StructType,
                                                    skipNonExistsCol: Boolean = false): StructType = {
    val (inputColNames, outputColNames) = getInOutCols()

    require(outputColNames.distinct.length == outputColNames.length,
      s"Output columns should not be duplicate.")

    val outputFields = inputColNames.zip(outputColNames).flatMap {
      case (inputColName, outputColName) =>
        schema.fieldNames.contains(inputColName) match {
          case true => Some(validateAndTransformField(schema, inputColName, outputColName))
          case false if skipNonExistsCol => None
          case _ => throw new SparkException(s"Input column $inputColName does not exist.")
        }
    }
    StructType(schema.fields ++ outputFields)
  }
}


@Since("1.4.0")
class LAMLStringIndexer @Since("1.4.0")(
                                               @Since("1.4.0") override val uid: String
                                       ) extends Estimator[LAMLStringIndexerModel]
        with LAMLStringIndexerBase with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("strIdx"))

  /** @group setParam */
  @Since("1.6.0")
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /** @group setParam */
  @Since("2.3.0")
  def setStringOrderType(value: String): this.type = set(stringOrderType, value)

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  @Since("3.0.0")
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  @Since("3.0.0")
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  // TODO: minFreqs required refactoring in multiple ways: 1 - parameter validation, 2 - default value, 3 - single minFreq
  @Since("3.2.0")
  val minFreqs: IntArrayParam = new IntArrayParam(this, "minFreqs", doc = "minFreqs")

  /** @group setParam */
  @Since("3.2.0")
  def setMinFreq(value: Array[Int]): this.type = set(minFreqs, value)

  @Since("3.2.0")
  val defaultValue: DoubleParam = new DoubleParam(this, "defaultValue", doc = "defaultValue")

  /** @group setParam */
  @Since("3.2.0")
  def setDefaultValue(value: Double): this.type = set(defaultValue, value)

  @Since("3.2.0")
  val freqLabel: BooleanParam = new BooleanParam(this, "freqLabel", doc = "freqLabel")

  /** @group setParam */
  @Since("3.2.0")
  def setFreqLabel(value: Boolean): this.type = set(freqLabel, value)

  @Since("3.2.0")
  val nanLast: BooleanParam = new BooleanParam(this, "nanLast", doc = "nanLast")

  /** @group setParam */
  @Since("3.2.0")
  def setNanLast(value: Boolean): this.type = set(nanLast, value)

  setDefault(defaultValue -> 0.0D, nanLast -> false, freqLabel -> false)

  private def getSelectedCols(dataset: Dataset[_], inputCols: Seq[String]): Seq[Column] = {
    inputCols.map { colName =>
      val col = dataset.col(colName)
      if (col.expr.dataType == StringType) {
        col
      } else if (col.expr.dataType == BooleanType) {
        //col.cast(StringType)
        new Column(col.expr).cast(StringType)
      } else {
        // We don't count for NaN values. Because `StringIndexerAggregator` only processes strings,
        // we replace NaNs with null in advance.
        new Column(If(col.isNaN.expr, Literal("NaN")/*Literal(null)*/, col.expr)).cast(StringType)
      }
    }
  }

  private def countByValue(dataset: Dataset[_],
                           inputCols: Array[String]): Array[OpenHashMap[String, Long]] = {

    val aggregator = new StringIndexerAggregator(inputCols.length)
    implicit val encoder: Encoder[Array[OpenHashMap[String, Long]]] = Encoders.kryo[Array[OpenHashMap[String, Long]]]

    val selectedCols = getSelectedCols(dataset, inputCols)
    dataset.select(selectedCols: _*)
            .toDF
            .groupBy().agg(aggregator.toColumn)
            .as[Array[OpenHashMap[String, Long]]]
            .collect()(0)
  }

  private def sortByFreq(dataset: Dataset[_], ascending: Boolean): Array[Array[(String, Long)]] = {
    val (inputCols, _) = getInOutCols()

    val sortFunc = LAMLStringIndexer.getSortFunc(ascending = ascending)
    val orgStrings = countByValue(dataset, inputCols).toSeq zip $(minFreqs)

    val arr = ThreadUtils.parmap(orgStrings, "sortingStringLabels", 8) { case (counts, minFreq) =>
      counts.toSeq.filter(_._2 > minFreq).sortWith(sortFunc).map(v => (v._1, v._2)).toArray
    }

    if ($(nanLast)){
      arr.map(v => v.filter(w => !w._1.equals("NaN")) :+ ("NaN", 0L)).toArray
    } else {
      arr.toArray
    }
  }

  private def sortByAlphabet(dataset: Dataset[_], ascending: Boolean): Array[Array[(String, Long)]] = {
    val (inputCols, _) = getInOutCols()

    val sortFunc = LAMLStringIndexer.getAlphabetSortFunc(ascending = ascending)
    val orgStrings = countByValue(dataset, inputCols).toSeq zip $(minFreqs)

    val arr = ThreadUtils.parmap(orgStrings, "sortingStringLabels", 8) { case (counts, minFreq) =>
      counts.toSeq.filter(_._2 > minFreq).sortWith(sortFunc).map(v => (v._1, v._2)).toArray
    }

    if ($(nanLast)){
      arr.map(v => v.filter(w => !w._1.equals("NaN")) :+ ("NaN", 0L)).toArray
    } else {
      arr.toArray
    }
  }

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): LAMLStringIndexerModel = {
    transformSchema(dataset.schema, logging = true)

    // In case of equal frequency when frequencyDesc/Asc, the strings are further sorted
    // alphabetically.
    val labelsArray = $(stringOrderType) match {
      case LAMLStringIndexer.frequencyDesc => sortByFreq(dataset, ascending = false)
      case LAMLStringIndexer.frequencyAsc => sortByFreq(dataset, ascending = true)
      case LAMLStringIndexer.alphabetDesc => sortByAlphabet(dataset, ascending = false)
      case LAMLStringIndexer.alphabetAsc => sortByAlphabet(dataset, ascending = true)
    }
    copyValues(
      new LAMLStringIndexerModel(
        uid = uid,
        labelsArray = labelsArray
      ).setDefaultValue($(defaultValue)).setFreqLabel($(freqLabel)).setNanLast($(nanLast)).setParent(this)
    )
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): LAMLStringIndexer = defaultCopy(extra)
}


@Since("1.6.0")
object LAMLStringIndexer extends DefaultParamsReadable[LAMLStringIndexer] {
  private[feature] val SKIP_INVALID: String = "skip"
  private[feature] val ERROR_INVALID: String = "error"
  private[feature] val KEEP_INVALID: String = "keep"
  private[feature] val supportedHandleInvalids: Array[String] =
    Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)
  private[feature] val frequencyDesc: String = "frequencyDesc"
  private[feature] val frequencyAsc: String = "frequencyAsc"
  private[feature] val alphabetDesc: String = "alphabetDesc"
  private[feature] val alphabetAsc: String = "alphabetAsc"
  private[feature] val supportedStringOrderType: Array[String] =
    Array(frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc)

  @Since("1.6.0")
  override def load(path: String): LAMLStringIndexer = super.load(path)

  // Returns a function used to sort strings by frequency (ascending or descending).
  // In case of equal frequency, it sorts strings by alphabet (ascending).
  private[lightautoml] def getSortFunc(ascending: Boolean): ((String, Long), (String, Long)) => Boolean = {
    if (ascending) {
      case ((strA: String, freqA: Long), (strB: String, freqB: Long)) =>
        if (freqA == freqB) {
          strA < strB
        } else {
          freqA < freqB
        }
    } else {
      case ((strA: String, freqA: Long), (strB: String, freqB: Long)) =>
        if (freqA == freqB) {
          strA < strB
        } else {
          freqA > freqB
        }
    }
  }

  private[lightautoml] def getAlphabetSortFunc(ascending: Boolean): ((String, Long), (String, Long)) => Boolean = {
    if (ascending) {
      case ((strA: String, freqA: Long), (strB: String, freqB: Long)) =>
        if (strA == strB) {
          freqA < freqB
        } else {
          strA < strB
        }
    } else {
      case ((strA: String, freqA: Long), (strB: String, freqB: Long)) =>
        if (strA == strB) {
          freqA < freqB
        } else {
          strA > strB
        }
    }
  }

}


@Since("1.4.0")
class LAMLStringIndexerModel(override val uid: String,
                             val labelsArray: Array[Array[(String, Long)]])
        extends Model[LAMLStringIndexerModel] with LAMLStringIndexerBase with MLWritable {

  import LAMLStringIndexerModel._

  @Since("1.5.0")
  def this(uid: String, labels: Array[(String, Long)]) = this(uid, Array(labels))

  @Since("1.5.0")
  def this(labels: Array[(String, Long)]) = this(Identifiable.randomUID("strIdx"), Array(labels))

  @Since("3.0.0")
  def this(labelsArray: Array[Array[(String, Long)]]) = this(Identifiable.randomUID("strIdx"), labelsArray)

  /** @group setParam */
  @Since("1.6.0")
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  @Since("3.0.0")
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  @Since("3.0.0")
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  @Since("3.2.0")
  val defaultValue: DoubleParam = new DoubleParam(this, "defaultValue", doc = "defaultValue")

  /** @group setParam */
  @Since("3.2.0")
  def setDefaultValue(value: Double): this.type = set(defaultValue, value)

  @Since("3.2.0")
  val freqLabel: BooleanParam = new BooleanParam(this, "freqLabel", doc = "freqLabel")

  /** @group setParam */
  @Since("3.2.0")
  def setFreqLabel(value: Boolean): this.type = set(freqLabel, value)

  @Since("3.2.0")
  val nanLast: BooleanParam = new BooleanParam(this, "nanLast", doc = "nanLast")

  /** @group setParam */
  @Since("3.2.0")
  def setNanLast(value: Boolean): this.type = set(nanLast, value)

  setDefault(defaultValue -> 0.0D, freqLabel -> false, nanLast -> false)

  @deprecated("`labels` is deprecated and will be removed in 3.1.0. Use `labelsArray` " +
          "instead.", "3.0.0")
  @Since("1.5.0")
  def labels: Array[Array[String]] = {
    require(labelsArray.length == 1, "This StringIndexerModel is fit on multiple columns. " +
            "Call `labelsArray` instead.")
    labelsArray(0).map(v => Array(v._1, v._2.toString))
  }

  @Since("3.2.0")
  def getStringLabels: Array[Array[Array[String]]] = {
    labelsArray.map((col: Array[(String, Long)]) => col.map(pair => Array(pair._1, pair._2.toString)))
  }

  // Prepares the maps for string values to corresponding index values.
  private lazy val labelsToIndexArray: Array[OpenHashMap[String, Double]] = {
    for (labels <- labelsArray) yield {
      val n = labels.length
      val map = new OpenHashMap[String, Double](n)
      if ($(freqLabel)) {
        labels.foreach { label => map.update(label._1, label._2) }
      } else {
        labels.zipWithIndex.foreach { case (label, idx) =>
          map.update(label._1, idx + 1)
        }
      }
      map
    }
  }

  // This filters out any null values and also the input labels which are not in
  // the dataset used for fitting.
  private def filterInvalidData(dataset: Dataset[_], inputColNames: Seq[String]): Dataset[_] = {
    val conditions: Seq[Column] = (0 until inputColNames.length).map { i =>
      val inputColName = inputColNames(i)
      val labelToIndex = labelsToIndexArray(i)
      // We have this additional lookup at `labelToIndex` when `handleInvalid` is set to
      // `StringIndexer.SKIP_INVALID`. Another idea is to do this lookup natively by SQL
      // expression, however, lookup for a key in a map is not efficient in SparkSQL now.
      // See `ElementAt` and `GetMapValue` expressions. If SQL's map lookup is improved,
      // we can consider to change this.
      val ctx = SparkContext.getActive.get
      val labelToIndexBcst = ctx.broadcast(labelToIndex)

      val filter = udf { label: String =>
        val l2idx = labelToIndexBcst.value
        l2idx.contains(label)
      }
      filter(dataset(inputColName))
    }

    dataset.na.drop(inputColNames.filter(dataset.schema.fieldNames.contains(_)))
            .where(conditions.reduce(_ and _))
  }

  private def getIndexer(labelToIndex: OpenHashMap[String, Double]) = {
    val keepInvalid = (getHandleInvalid == StringIndexer.KEEP_INVALID)

    val ctx = SparkContext.getActive.get
    val labelToIndexBcst = ctx.broadcast(labelToIndex)

    val isNanLast = $(nanLast)
    val defaultVal = $(defaultValue)

    udf { label: String =>
      val l2idx = labelToIndexBcst.value

      if (label == null) {
        if (keepInvalid) {
          if (isNanLast){
            l2idx("NaN")
          } else {
            defaultVal
          }
        } else {
          throw new SparkException("StringIndexer encountered NULL value. To handle or skip " +
                  "NULLS, try setting StringIndexer.handleInvalid.")
        }
      } else {
        if (l2idx.contains(label)) {
          l2idx(label)
        } else if (keepInvalid) {
          defaultVal
        } else {
          throw new SparkException(s"Unseen label: $label. To handle unseen labels, " +
                  s"set Param handleInvalid to ${StringIndexer.KEEP_INVALID}.")
        }
      }
    }.asNondeterministic()
  }

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val (inputColNames, outputColNames) = getInOutCols()
    val outputColumns = new Array[Column](outputColNames.length)

    // Skips invalid rows if `handleInvalid` is set to `StringIndexer.SKIP_INVALID`.
    val filteredDataset = if (getHandleInvalid == StringIndexer.SKIP_INVALID) {
      filterInvalidData(dataset, inputColNames)
    } else {
      dataset
    }

    for (i <- outputColNames.indices) {
      val inputColName = inputColNames(i)
      val outputColName = outputColNames(i)
      val labelToIndex = labelsToIndexArray(i)
      val labels = labelsArray(i)

      if (!dataset.schema.fieldNames.contains(inputColName)) {
        logWarning(s"Input column ${inputColName} does not exist during transformation. " +
                "Skip StringIndexerModel for this column.")
        outputColNames(i) = null
      } else {

        // we don't put labels themselves into metadata
        // cause they can weigh far too much (for instance, 100 - 500 MB)
        // and would be broadcasted with the task each time
        // which creates additional overheads on deserialization of tasks in workers
        val metadata = NominalAttribute.defaultAttr
                .withName(outputColName)
                .withNumValues(labels.length)
                .toMetadata()

        val indexer = getIndexer(labelToIndex)

        outputColumns(i) = indexer(dataset(inputColName).cast(StringType))
                .as(outputColName, metadata)
      }
    }

    val filteredOutputColNames = outputColNames.filter(_ != null)
    val filteredOutputColumns = outputColumns.filter(_ != null)

    require(filteredOutputColNames.length == filteredOutputColumns.length)
    if (filteredOutputColNames.length > 0) {
      filteredDataset.withColumns(filteredOutputColNames, filteredOutputColumns)
    } else {
      filteredDataset.toDF()
    }
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, skipNonExistsCol = true)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): LAMLStringIndexerModel = {
    val copied = new LAMLStringIndexerModel(uid, labelsArray)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: LAMLStringIndexModelWriter = new LAMLStringIndexModelWriter(this)

  @Since("3.0.0")
  override def toString: String = {
    s"StringIndexerModel: uid=$uid, handleInvalid=${$(handleInvalid)}" +
            get(stringOrderType).map(t => s", stringOrderType=$t").getOrElse("") +
            get(inputCols).map(c => s", numInputCols=${c.length}").getOrElse("") +
            get(outputCols).map(c => s", numOutputCols=${c.length}").getOrElse("")
  }
}

@Since("1.6.0")
object LAMLStringIndexerModel extends MLReadable[LAMLStringIndexerModel] {

  private[LAMLStringIndexerModel]
  class LAMLStringIndexModelWriter(instance: LAMLStringIndexerModel) extends MLWriter {

    private case class Data(labelsArray: Array[Array[(String, Long)]])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.labelsArray)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class LAMLStringIndexerModelReader extends MLReader[LAMLStringIndexerModel] {

    private val className = classOf[LAMLStringIndexerModel].getName

    override def load(path: String): LAMLStringIndexerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString

      // We support loading old `StringIndexerModel` saved by previous Spark versions.
      // Previous model has `labels`, but new model has `labelsArray`.
      val (majorVersion, minorVersion) = majorMinorVersion(metadata.sparkVersion)
      val labelsArray = if (majorVersion < 3) {
        // Spark 2.4 and before.
        val data = sparkSession.read.parquet(dataPath)
                .select("labels")
                .head()
        val labels = data.getAs[Seq[(String, Long)]](0).toArray
        Array(labels)
      } else {
        // After Spark 3.0.
        val data = sparkSession.read.parquet(dataPath)
                .select("labelsArray")
                .head()

        val res = data.getSeq[scala.collection.Seq[GenericRowWithSchema]](0)
        res.map(_.map(x => (x.getAs[String](0), x.getAs[Long](1))).toArray).toArray
      }
      val model = new LAMLStringIndexerModel(metadata.uid, labelsArray)
      metadata.getAndSetParams(model)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[LAMLStringIndexerModel] = new LAMLStringIndexerModelReader

  @Since("1.6.0")
  override def load(path: String): LAMLStringIndexerModel = super.load(path)
}
