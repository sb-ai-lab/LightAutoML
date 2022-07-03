.. role:: hidden
    :class: hidden-section

lightautoml.spark.automl
======================

The main module, which includes the SparkAutoML class, blenders and ready-made presets.

.. currentmodule:: lightautoml.spark.automl.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkAutoML


Presets
-------

Presets for end-to-end model training for special tasks.

.. currentmodule:: lightautoml.spark.automl.presets

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    base.SparkAutoMLPreset
    tabular_presets.SparkTabularAutoML


Blenders
--------

.. currentmodule:: lightautoml.spark.automl.blend

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    SparkBlender
    SparkBestModelSelector
    SparkMeanBlender
    SparkWeightedBlender
