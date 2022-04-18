.. role:: hidden
    :class: hidden-section


lightautoml.validation
==============================

The module provide classes and functions for model validation.

Iterators
------------------------------

.. currentmodule:: lightautoml.validation

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~base.TrainValidIterator
    ~base.DummyIterator
    ~base.HoldoutIterator
    ~base.CustomIterator
    ~np_iterators.FoldsIterator
    ~np_iterators.TimeSeriesIterator


Iterators Getters and Utils
------------------------------


.. currentmodule:: lightautoml.validation

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    ~utils.create_validation_iterator
    ~np_iterators.get_numpy_iterator
