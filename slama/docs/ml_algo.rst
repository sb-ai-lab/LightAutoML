.. role:: hidden
    :class: hidden-section

lightautoml.ml_algo
===================

Models used for machine learning pipelines.

Base Classes
------------------------

.. currentmodule:: lightautoml.ml_algo.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    MLAlgo
    TabularMLAlgo


Linear Models
-------------------------

.. currentmodule:: lightautoml.ml_algo

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~linear_sklearn.LinearLBFGS
    ~linear_sklearn.LinearL1CD
    ~dl_model.TorchModel

Boosted Trees
-------------------------

.. currentmodule:: lightautoml.ml_algo

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~boost_lgbm.BoostLGBM
    ~boost_cb.BoostCB

WhiteBox
-------------------------

.. currentmodule:: lightautoml.ml_algo

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~whitebox.WbMLAlgo
