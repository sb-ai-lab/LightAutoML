.. role:: hidden
    :class: hidden-section


lightautoml.tasks.losses
==============================

Wrappers of loss and metric functions for different machine learning algorithms.

Base Classes
------------

.. currentmodule:: lightautoml.tasks.losses.base

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    MetricFunc
    Loss


Wrappers for LightGBM
---------------------

Classes
^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~lgb.LGBFunc
    ~lgb.LGBLoss

Functions
^^^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    ~lgb_custom.softmax_ax1
    ~lgb_custom.lgb_f1_loss_multiclass



Wrappers for CatBoost
---------------------

Classes
^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~cb.CBLoss
    ~cb_custom.CBCustomMetric
    ~cb_custom.CBRegressionMetric
    ~cb_custom.CBClassificationMetric
    ~cb_custom.CBMulticlassMetric


Functions
^^^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    ~cb.cb_str_loss_wrapper


Wrappers for Sklearn
---------------------

Classes
^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~sklearn.SKLoss


Wrappers for Torch
---------------------

Classes
^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~torch.TorchLossWrapper
    ~torch.TORCHLoss


Functions
^^^^^^^^^

.. currentmodule:: lightautoml.tasks.losses

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    ~torch.torch_rmsle
    ~torch.torch_quantile
    ~torch.torch_fair
    ~torch.torch_huber
    ~torch.torch_f1
    ~torch.torch_mape
