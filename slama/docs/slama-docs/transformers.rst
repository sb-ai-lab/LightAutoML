.. role:: hidden
    :class: hidden-section


lightautoml.spark.transformers
==============================

Basic feature generation steps and helper utils.

Base Classes
------------------------------

.. currentmodule:: lightautoml.spark.transformers.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkBaseEstimator
    SparkBaseTransformer
    SparkChangeRolesTransformer
    SparkSequentialTransformer
    SparkUnionTransformer
    SparkColumnsAndRoles
    HasInputRoles
    HasOutputRoles
    ColumnsSelectorTransformer
    DropColumnsTransformer
    PredictionColsTransformer
    ProbabilityColsTransformer



Numeric
------------------------------

.. currentmodule:: lightautoml.spark.transformers.numeric

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkFillnaMedianEstimator
    SparkNaNFlagsEstimator
    SparkQuantileBinningEstimator
    SparkStandardScalerEstimator
    SparkFillInfTransformer
    SparkFillnaMedianTransformer
    SparkLogOddsTransformer
    SparkNaNFlagsTransformer
    SparkQuantileBinningTransformer
    SparkStandardScalerTransformer



Categorical
------------------------------

.. currentmodule:: lightautoml.spark.transformers.categorical

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkLabelEncoderEstimator
    SparkOrdinalEncoderEstimator
    SparkFreqEncoderEstimator
    SparkCatIntersectionsEstimator
    SparkTargetEncoderEstimator
    SparkMulticlassTargetEncoderEstimator
    SparkOHEEncoderEstimator
    SparkLabelEncoderTransformer
    SparkOrdinalEncoderTransformer
    SparkFreqEncoderTransformer
    SparkCatIntersectionsTransformer
    SparkTargetEncoderTransformer
    SparkMultiTargetEncoderTransformer
    SparkCatIntersectionsHelper

Categorical (Scala)
------------------------------

.. currentmodule:: lightautoml.spark.transformers.scala_wrappers

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    laml_string_indexer.LAMLStringIndexer
    laml_string_indexer.LAMLStringIndexerModel


Datetime
------------------------------

.. currentmodule:: lightautoml.spark.transformers.datetime

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkTimeToNumTransformer
    SparkBaseDiffTransformer
    SparkDateSeasonsTransformer
    SparkDatetimeHelper