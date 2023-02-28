.. role:: hidden
    :class: hidden-section


lightautoml.reader
=====================

Utils for reading, training and analysing data.

Readers
-------------

.. currentmodule:: lightautoml.reader.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    Reader
    PandasToPandasReader


Tabular Batch Generators
-----------------------------

Batch Handler Classes
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: lightautoml.reader.tabular_batch_generator

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    Batch
    FileBatch
    BatchGenerator
    DfBatchGenerator
    FileBatchGenerator

Data Read Functions
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: lightautoml.reader.tabular_batch_generator

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    read_batch
    read_data
