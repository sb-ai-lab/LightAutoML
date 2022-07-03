.. role:: hidden
    :class: hidden-section


lightautoml.pipelines.selection
===============================

Feature selection module for ML pipelines.

Base Classes
-----------------

.. currentmodule:: lightautoml.pipelines.selection.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ImportanceEstimator
    SelectionPipeline

Importance Based Selectors
--------------------------

.. currentmodule:: lightautoml.pipelines.selection

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~importance_based.ModelBasedImportanceEstimator
    ~importance_based.ImportanceCutoffSelector
    ~permutation_importance_based.NpPermutationImportanceEstimator
    ~permutation_importance_based.NpIterativeFeatureSelector

Other Selectors
----------------------

.. currentmodule:: lightautoml.pipelines.selection

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~linear_selector.HighCorrRemoval
