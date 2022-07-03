.. role:: hidden
    :class: hidden-section


lightautoml.transformers
==============================

Basic feature generation steps and helper utils.

Base Classes
------------------------------

.. currentmodule:: lightautoml.transformers.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    LAMLTransformer
    SequentialTransformer
    UnionTransformer
    ColumnsSelector
    ColumnwiseUnion
    BestOfTransformers
    ConvertDataset
    ChangeRoles


Numeric
------------------------------

.. currentmodule:: lightautoml.transformers.numeric

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    NaNFlags
    FillnaMedian
    FillInf
    LogOdds
    StandardScaler
    QuantileBinning


Categorical
------------------------------

.. currentmodule:: lightautoml.transformers.categorical

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    LabelEncoder
    OHEEncoder
    FreqEncoder
    OrdinalEncoder
    TargetEncoder
    MultiClassTargetEncoder
    CatIntersectstions


Datetime
------------------------------

.. currentmodule:: lightautoml.transformers.datetime

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    TimeToNum
    BaseDiff
    DateSeasons


Decompositions
------------------------------

.. currentmodule:: lightautoml.transformers.decomposition

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    PCATransformer
    SVDTransformer


Text
------------------------------

.. currentmodule:: lightautoml.transformers.text

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    TunableTransformer
    TfidfTextTransformer
    TokenizerTransformer
    OneToOneTransformer
    ConcatTextTransformer
    AutoNLPWrap


Image
------------------------------

.. currentmodule:: lightautoml.transformers.image

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ImageFeaturesTransformer
    AutoCVWrap
