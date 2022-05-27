from pyspark import since, keyword_only, SparkContext
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, \
    HasInputCols, HasOutputCols, HasHandleInvalid, TypeConverters, Param, Params
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams
from pyspark.ml.common import inherit_doc

from lightautoml.spark.mlwriters import LAMLStringIndexerModelJavaMLReadable


class _StringIndexerParams(JavaParams, HasHandleInvalid, HasInputCol, HasOutputCol,
                           HasInputCols, HasOutputCols):
    """
    Params for :py:class:`StringIndexer` and :py:class:`StringIndexerModel`.
    """

    stringOrderType = Param(Params._dummy(), "stringOrderType",
                            "How to order labels of string column. The first label after " +
                            "ordering is assigned an index of 0. Supported options: " +
                            "frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc. " +
                            "Default is frequencyDesc. In case of equal frequency when " +
                            "under frequencyDesc/Asc, the strings are further sorted " +
                            "alphabetically",
                            typeConverter=TypeConverters.toString)

    handleInvalid = Param(Params._dummy(), "handleInvalid", "how to handle invalid data (unseen " +
                          "or NULL values) in features and label column of string type. " +
                          "Options are 'skip' (filter out rows with invalid data), " +
                          "error (throw an error), or 'keep' (put invalid data " +
                          "in a special additional bucket, at index numLabels).",
                          typeConverter=TypeConverters.toString)

    minFreqs = Param(Params._dummy(),
                     "minFreqs",
                     "The minimum number of the element occurrences not to skip it.",
                     typeConverter=TypeConverters.toListInt)

    defaultValue = Param(Params._dummy(),
                         "defaultValue",
                         "The index for unknown labels and labels that are too rare.",
                         typeConverter=TypeConverters.toFloat)

    freqLabel = Param(Params._dummy(),
                      "freqLabel",
                      "If true, label will be transformed to its occurrences",
                      typeConverter=TypeConverters.toBoolean)

    nanLast = Param(Params._dummy(),
                    "nanLast",
                    "If true, appends 'NaN' label to the end of each mapping",
                    typeConverter=TypeConverters.toBoolean)

    def __init__(self, *args):
        super(_StringIndexerParams, self).__init__(*args)
        self._setDefault(
            handleInvalid="error",
            stringOrderType="frequencyDesc",
            minFreqs=tuple(),
            defaultValue=0.,
            freqLabel=False,
            nanLast=False
        )

    @since("2.3.0")
    def getStringOrderType(self):
        """
        Gets the value of :py:attr:`stringOrderType` or its default value 'frequencyDesc'.
        """
        return self.getOrDefault(self.stringOrderType)


class _StringIndexerModelParams(JavaParams, HasHandleInvalid, HasInputCol, HasOutputCol,
                                HasInputCols, HasOutputCols):
    """
    Params for :py:class:`StringIndexer` and :py:class:`StringIndexerModel`.
    """

    # stringOrderType = Param(Params._dummy(), "stringOrderType",
    #                         "How to order labels of string column. The first label after " +
    #                         "ordering is assigned an index of 0. Supported options: " +
    #                         "frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc. " +
    #                         "Default is frequencyDesc. In case of equal frequency when " +
    #                         "under frequencyDesc/Asc, the strings are further sorted " +
    #                         "alphabetically",
    #                         typeConverter=TypeConverters.toString)

    handleInvalid = Param(Params._dummy(), "handleInvalid", "how to handle invalid data (unseen " +
                          "or NULL values) in features and label column of string type. " +
                          "Options are 'skip' (filter out rows with invalid data), " +
                          "error (throw an error), or 'keep' (put invalid data " +
                          "in a special additional bucket, at index numLabels).",
                          typeConverter=TypeConverters.toString)

    # minFreqs = Param(Params._dummy(),
    #                  "minFreqs",
    #                  "The minimum number of the element occurrences not to skip it.",
    #                  typeConverter=TypeConverters.toListInt)

    defaultValue = Param(Params._dummy(),
                         "defaultValue",
                         "The index for unknown labels and labels that are too rare.",
                         typeConverter=TypeConverters.toFloat)

    freqLabel = Param(Params._dummy(),
                      "freqLabel",
                      "If true, label will be transformed to its occurrences",
                      typeConverter=TypeConverters.toBoolean)

    nanLast = Param(Params._dummy(),
                    "nanLast",
                    "If true, appends 'NaN' label to the end of each mapping",
                    typeConverter=TypeConverters.toBoolean)

    def __init__(self, *args):
        super(_StringIndexerModelParams, self).__init__(*args)
        self._setDefault(
            handleInvalid="error",
            defaultValue=0.,
            freqLabel=False,
            nanLast=False
        )


@inherit_doc
class LAMLStringIndexer(JavaEstimator, _StringIndexerParams, JavaMLReadable, JavaMLWritable):
    """
    Custom implementation of PySpark StringIndexer wrapper
    """

    @keyword_only
    def __init__(self, *, inputCol=None, outputCol=None, inputCols=None, outputCols=None,
                 handleInvalid="error", stringOrderType="frequencyDesc", minFreqs=None,
                 defaultValue=0., freqLabel=False, nanLast=False):
        """
        __init__(self, \\*, inputCol=None, outputCol=None, inputCols=None, outputCols=None, \
                 handleInvalid="error", stringOrderType="frequencyDesc")
        """
        super(LAMLStringIndexer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexer",
            self.uid
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("1.4.0")
    def setParams(self, *, inputCol=None, outputCol=None, inputCols=None, outputCols=None,
                  handleInvalid="error", stringOrderType="frequencyDesc", minFreqs=None,
                  defaultValue=0., freqLabel=False, nanLast=False):

        """
        setParams(self, \\*, inputCol=None, outputCol=None, inputCols=None, outputCols=None, \
                  handleInvalid="error", stringOrderType="frequencyDesc")
        Sets params for this StringIndexer.
        """

        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return LAMLStringIndexerModel(java_model)

    @since("2.3.0")
    def setStringOrderType(self, value):
        """
        Sets the value of :py:attr:`stringOrderType`.
        """
        return self._set(stringOrderType=value)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    @since("3.0.0")
    def setInputCols(self, value):
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    @since("3.0.0")
    def setOutputCols(self, value):
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self._set(outputCols=value)

    def setHandleInvalid(self, value):
        """
        Sets the value of :py:attr:`handleInvalid`.
        """
        return self._set(handleInvalid=value)

    @since("3.2.0")
    def setMinFreqs(self, value):
        """
        Sets the value of :py:attr:`minFreqs`.
        """
        return self._set(minFreqs=value)

    @since("3.2.0")
    def setDefaultValue(self, value):
        """
        Sets the value of :py:attr:`defaultValue`.
        """
        return self._set(defaultValue=value)

    @since("3.2.0")
    def setFreqLabel(self, value):
        """
        Sets the value of :py:attr:`freqLabel`.
        """
        return self._set(freqLabel=value)

    @since("3.2.0")
    def setNanLast(self, value):
        """
        Sets the value of :py:attr:`nanLast`.
        """
        return self._set(nanLast=value)


class LAMLStringIndexerModel(JavaModel, _StringIndexerModelParams, LAMLStringIndexerModelJavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`StringIndexer`.

    .. versionadded:: 1.4.0
    """

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    @since("3.0.0")
    def setInputCols(self, value):
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    @since("3.0.0")
    def setOutputCols(self, value):
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self._set(outputCols=value)

    @since("2.4.0")
    def setHandleInvalid(self, value):
        """
        Sets the value of :py:attr:`handleInvalid`.
        """
        return self._set(handleInvalid=value)

    @since("3.2.0")
    def setDefaultValue(self, value):
        """
        Sets the value of :py:attr:`defaultValue`.
        """
        return self._set(defaultValue=value)

    @since("3.2.0")
    def setFreqLabel(self, value):
        """
        Sets the value of :py:attr:`freqLabel`.
        """
        return self._set(freqLabel=value)

    @since("3.2.0")
    def setNanLast(self, value):
        """
        Sets the value of :py:attr:`nanLast`.
        """
        return self._set(nanLast=value)

    @classmethod
    @since("2.4.0")
    def from_labels(cls, labels, inputCol, outputCol=None, handleInvalid=None,
                    defaultValue=0., freqLabel=False, nanLast=False):
        """
        Construct the model directly from an array of label strings,
        requires an active SparkContext.
        """
        sc = SparkContext._active_spark_context
        java_class = sc._gateway.jvm.java.lang.String
        jlabels = LAMLStringIndexerModel._new_java_array(labels, java_class)
        model = LAMLStringIndexerModel._create_from_java_class(
            "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexerModel",
            jlabels
        )
        model.setInputCol(inputCol)
        if outputCol is not None:
            model.setOutputCol(outputCol)
        if handleInvalid is not None:
            model.setHandleInvalid(handleInvalid)

        model.setDefaultValue(defaultValue)
        model.setFreqLabel(freqLabel)
        model.setNanLast(nanLast)

        return model

    @classmethod
    @since("3.0.0")
    def from_arrays_of_labels(cls, arrayOfLabels, inputCols, outputCols=None,
                              handleInvalid=None, defaultValue=0., freqLabel=False):
        """
        Construct the model directly from an array of array of label strings,
        requires an active SparkContext.
        """
        sc = SparkContext._active_spark_context
        java_class = sc._gateway.jvm.java.lang.String
        jlabels = LAMLStringIndexerModel._new_java_array(arrayOfLabels, java_class)
        model = LAMLStringIndexerModel._create_from_java_class(
            "org.apache.spark.ml.feature.lightautoml.LAMLStringIndexerModel",
            jlabels
        )
        model.setInputCols(inputCols)
        if outputCols is not None:
            model.setOutputCols(outputCols)
        if handleInvalid is not None:
            model.setHandleInvalid(handleInvalid)

        model.setDefaultValue(defaultValue)
        model.setFreqLabel(freqLabel)

        return model

    @property
    @since("1.5.0")
    def labels(self):
        """
        Ordered list of labels, corresponding to indices to be assigned.

        .. deprecated:: 3.1.0
            It will be removed in future versions. Use `labelsArray` method instead.
        """
        return self._call_java("labels")

    @property
    @since("3.0.2")
    def labelsArray(self):
        """
        Array of ordered list of labels, corresponding to indices to be assigned
        for each input column.
        """
        return self._call_java("getStringLabels")

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.

        Meta-algorithms such as Pipeline should override this method as a classmethod.
        """

        def __get_class(clazz):
            """
            Loads Python class from its name.
            """
            parts = clazz.split(".")
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m

        stage_name = "lightautoml.spark.transformers.scala_wrappers.laml_string_indexer.LAMLStringIndexerModel"
        # Generate a default new instance from the stage_name class.
        py_type = __get_class(stage_name)
        if issubclass(py_type, JavaParams):
            # Load information from java_stage to the instance.
            py_stage = py_type()
            py_stage._java_obj = java_stage
            py_stage._resetUid(java_stage.uid())
            py_stage._transfer_params_from_java()
        elif hasattr(py_type, "_from_java"):
            py_stage = py_type._from_java(java_stage)
        else:
            raise NotImplementedError("This Java stage cannot be loaded into Python currently: %r" % stage_name)
        return py_stage
