.. role:: hidden
    :class: hidden-section


lightautoml.tasks
==============================

See :doc:`../Advanced_metrics` for the metric catalog, parameters and examples.

.. toctree::
    :hidden:

    ../Advanced_metrics

Task Class
----------

.. currentmodule:: lightautoml.tasks.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    Task




Common Metrics
-----------------------

Classes
^^^^^^^^^^^

.. currentmodule:: lightautoml.tasks.common_metric

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    F1Factory
    BestClassBinaryWrapper
    BestClassMulticlassWrapper


Functions
^^^^^^^^^^

.. currentmodule:: lightautoml.tasks.common_metric

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    mean_quantile_error
    mean_huber_error
    mean_fair_error
    mean_absolute_percentage_error
    roc_auc_ovr
    rmsle
    auc_mu


Metric lookup
-------------

.. autofunction:: lightautoml.tasks.metrics.get_metric
