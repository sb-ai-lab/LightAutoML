Frequently Asked Questions
==========================

This page explains behavior that is easy to mistake for a bug. The answers
target the current stable LightAutoML release. If an example from an old
discussion behaves differently, first check the installed versions of
LightAutoML, Python, pandas, scikit-learn, LightGBM, and XGBoost.


Training and validation
-----------------------

What does ``fit_predict`` return?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without ``valid_data``, ``fit_predict`` returns out-of-fold (OOF)
predictions for the training rows. Each row is predicted by a fold model that
was not trained on that row.

With ``valid_data``, it returns predictions for that validation dataset
instead. In both cases, call ``.data`` on the returned dataset to get the
underlying NumPy array.

OOF predictions are not the same as out-of-bag (OOB) predictions. OOF comes
from cross-validation. OOB is a bagging concept: it uses samples that were
not selected into a particular bootstrap sample.

Why should I evaluate the training data with OOF predictions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling ``predict(train_data)`` evaluates fitted models on rows they may have
seen during training, so the result is optimistic. Use the value returned by
``fit_predict`` for a less biased training-set estimate.

Keep this value if it is needed later: OOF predictions are returned by
``fit_predict`` but are not stored as part of the serialized AutoML model.

Why are some OOF predictions missing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time limit may expire before every fold or algorithm finishes. Inspect
the training log and increase ``timeout`` when incomplete OOF predictions
are caused by unfinished folds.

The small time limits used in tutorials are intended to make examples finish
quickly. They are not recommended production defaults. Uplift, neural
networks, tuning, nested cross-validation, and utilized presets can require
substantially more time.

Does a large ``timeout`` force LightAutoML to use all that time?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. ``timeout`` is an upper bound. ``TabularAutoML`` can finish early after
all configured pipelines complete. Use a utilized preset or increase tuning
budgets when the goal is to explore more configurations.

How do I use a fixed validation set?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass it as ``valid_data`` to ``fit_predict``. This replaces the ordinary
outer cross-validation prediction with holdout prediction. Multi-level
stacking requires CV folds and cannot be fitted with ``valid_data``.


Targets and predictions
-----------------------

Why does the order of class-probability columns differ from my labels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For classification targets that are not already encoded as ``0..K-1``,
LightAutoML can map labels to internal integers. Inspect
``automl.reader.class_mapping`` to recover the mapping. It is ``None`` when
no remapping was required.

For multiclass prediction, output columns follow this internal class order.
Do not infer column meaning from alphabetical order or from the original
label values.

What does ``return_all_predictions=True`` change?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, prediction returns the final blended output. With
``return_all_predictions=True``, it returns predictions from all models in
the last level. For multiclass tasks, each model contributes one block of
class-probability columns in the class order described above.

Why is an algorithm absent from the final ensemble?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weighted blender can prune models whose optimized weight is below
``weighted_blender_max_nonzero_coef``. A model may therefore train
successfully but not appear in the final ensemble.

Why are binary probabilities concentrated near the class rate?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can be normal for a weakly separating model trained with log loss. A
probability range narrower than ``0..1`` is not itself evidence of a bug, and
``0.5`` is not automatically the best decision threshold for an imbalanced
problem. Choose the threshold on validation data for the required metric or
business cost. Probability calibration is a separate step.


Features and roles
------------------

Why did LightAutoML treat a numeric column as categorical?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``reader_params={"advanced_roles": True}``, the reader can choose a
processing role from data statistics rather than only from the pandas dtype.
Specify roles explicitly when domain knowledge should override this choice,
or disable advanced role inference.

Why is a text column not processed as NLP?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An unspecified string column in ``TabularAutoML`` is normally treated as a
categorical feature. Assign a text role and use the appropriate NLP preset or
build a custom pipeline when TF-IDF or transformer features are required.

Does feature selection affect every model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not necessarily. ``selection_params["select_algos"]`` controls which
algorithm families receive selected features. In the default tabular
configuration it contains ``"gbm"``; linear and random-forest pipelines do
not automatically receive the same selected subset.

``automl.collect_used_feats()`` returns the features retained by the fitted
AutoML object. Fast feature importance should not be interpreted as proof
that every lower-ranked feature is safe to remove.


Configuration and resources
---------------------------

How is ``general_params["use_algos"]`` structured?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is either ``"auto"`` or a list of levels. For example,
``[["lgb", "cb"], ["linear_l2"]]`` defines two algorithms on the first level
and one on the second. Available names depend on the installed LightAutoML
version; use the current preset configuration as the source of truth instead
of copying names from old chat messages.

The current tabular preset supports tuned neural-network names such as
``"nn_tuned"``. Older releases may not support the same list.

What do ``cpu_limit`` and ``memory_limit`` mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

They are resource constraints passed to the preset, not measurements of all
available system resources. Some underlying libraries also allocate memory
or worker processes. Parallel inference via ``predict(..., n_jobs=N)`` can
use more RAM; set ``batch_size`` to trade speed for lower memory use.


Saving and loading
------------------

How do I save a fitted model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``path_to_save="model.joblib"`` to ``fit_predict``:

.. code-block:: python

    automl.fit_predict(
        train_data,
        roles={"target": "target"},
        path_to_save="model.joblib",
    )

    from joblib import load

    automl = load("model.joblib")

Alternatively, call ``joblib.dump(automl, "model.joblib")`` after fitting.
Serialize the whole fitted AutoML object, not individual internal estimators.

Can I load a model created by a different LightAutoML version?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Serialization is not a stable cross-version interchange format. A model can
also depend on the exact versions of pandas, scikit-learn, LightGBM,
CatBoost, XGBoost, PyTorch, and custom classes used during construction.
For production, preserve the training environment. Across significant
dependency or LightAutoML upgrades, retraining is safer than assuming that
an old joblib file is compatible.


Time series
-----------

Which roles are required for AutoTS?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify ``target`` and normally a date column using ``DatetimeRole``. For
global modelling across several series, also specify an ``id`` role.
Additional exogenous columns do not need special roles merely because they
are exogenous, but their values for date *t* must be information available
at date *t* to avoid leakage.

Why does AutoTS report ``need at least one array to concatenate``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One common cause is that no valid training sample can be built from the
requested history and forecast horizon. AutoTS uses regular portions of a
series; gaps in timestamps split that history. Check timestamp regularity,
missing dates, series length, history, and horizon before treating this as a
library defect.


Before reporting a bug
----------------------

Include:

* LightAutoML and Python versions;
* pandas, NumPy, scikit-learn, LightGBM, CatBoost, XGBoost, and PyTorch
  versions when relevant;
* task, preset, ``roles``, ``use_algos``, ``timeout``, and resource limits;
* dataset shape and a minimal schema without sensitive values;
* the complete traceback and the preceding LightAutoML log.

Check the `PyPI release history
<https://pypi.org/project/LightAutoML/#history>`_ before applying a workaround
from an old discussion. Compatibility issues reported against ``0.3.x`` may
already be fixed in ``0.4.x``.
