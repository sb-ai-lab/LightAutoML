.. role:: hidden
    :class: hidden-section

lightautoml.automl
======================

The main module, which includes the AutoML class, blenders and ready-made presets.

.. currentmodule:: lightautoml.automl.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    AutoML


Presets
-------

Presets for end-to-end model training for special tasks.

.. currentmodule:: lightautoml.automl.presets

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    base.AutoMLPreset
    tabular_presets.TabularAutoML
    tabular_presets.TabularUtilizedAutoML
    image_presets.TabularCVAutoML
    text_presets.TabularNLPAutoML
    whitebox_presets.WhiteBoxPreset


Blenders
--------

.. currentmodule:: lightautoml.automl.blend

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    Blender
    BestModelSelector
    MeanBlender
    WeightedBlender
