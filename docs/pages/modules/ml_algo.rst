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
    ~boost_xgb.BoostXGB

Training and tuning behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An exception in one Optuna trial marks that trial as failed and allows the
remaining trials to run. If ``fail_tolerance`` is exceeded, tuning stops and
uses the best completed trial's parameters. If no trial completes, training
uses the model's initial parameters. Invalid search-space distribution types
are rejected before tuning; keyboard interrupts still propagate. Failures
caused by invalid data or initial parameters can also affect the final fit.

XGBoost predictions and feature importance use ``best_iteration`` when it is
available, including when training reaches its iteration limit. Unused
features receive zero importance and retain their original names. Prediction
with ``booster="gblinear"`` uses the final coefficients because tree iteration
ranges do not apply to linear boosters.

For multiclass training, probability columns cover the encoded classes in
both training and validation data. CatBoost keeps these columns when a class
is absent from a training split. Cross-entropy scoring includes all probability
columns even when the validation labels contain only a subset of the classes.

CatBoost accepts numeric ``CSRSparseDataset`` inputs directly, including
missing values, without allocating a dense feature matrix. Encoded categorical
features in pandas datasets remain supported.

Custom LightGBM objectives are supported through both the older ``fobj`` API
and LightGBM 4's ``objective`` parameter. Explicit loss-adapter dictionaries
can be passed to ``Task``. The built-in multiclass F1 objective preserves the
derivative shape expected by the backend and applies observation weights.

XGBoost's ``rmsle`` loss trains squared error on ``log1p(target)`` and restores
predictions with ``expm1``. RMSLE evaluation uses RMSE in log space; other
requested metrics are evaluated on the original target scale. Quantile
parameter ``q`` and Huber parameter ``a`` are translated to the corresponding
XGBoost parameters. Quantile evaluation uses the requested quantile and
observation weights.


Neural Networks
-------------------------

.. currentmodule:: lightautoml.ml_algo.torch_based

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~nn_models.MLP
    ~nn_models.DenseLightModel
    ~nn_models.DenseModel
    ~nn_models.ResNetModel
    ~nn_models.SNN


WhiteBox
-------------------------

.. currentmodule:: lightautoml.ml_algo

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~whitebox.WbMLAlgo
