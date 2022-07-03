.. role:: hidden
    :class: hidden-section


lightautoml.pipelines.features
==============================

Pipelines for features generation.

Base Classes
-----------------

.. currentmodule:: lightautoml.pipelines.features.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    FeaturesPipeline
    EmptyFeaturePipeline
    TabularDataFeatures



Feature Pipelines for Boosting Models
-----------------------------------------

.. currentmodule:: lightautoml.pipelines.features.lgb_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    LGBSimpleFeatures
    LGBAdvancedPipeline


Feature Pipelines for Linear Models
-----------------------------------

.. currentmodule:: lightautoml.pipelines.features.linear_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    LinearFeatures

Feature Pipelines for WhiteBox
------------------------------

.. currentmodule:: lightautoml.pipelines.features.wb_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    WBFeatures


Image Feature Pipelines
----------------------------------

.. currentmodule:: lightautoml.pipelines.features.image_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ImageDataFeatures
    ImageSimpleFeatures
    ImageAutoFeatures


Text Feature Pipelines
------------------------------

.. currentmodule:: lightautoml.pipelines.features.text_pipeline

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    NLPDataFeatures
    TextAutoFeatures
    NLPTFiDFFeatures
    TextBertFeatures
