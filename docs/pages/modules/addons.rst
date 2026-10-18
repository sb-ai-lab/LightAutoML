.. role:: hidden
    :class: hidden-section

lightautoml.addons
==================

Extensions of core functionality.

Utilization
-----------

.. currentmodule:: lightautoml.addons.utilization

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~utilization.TimeUtilization

HypEx -- Hypothesises and Experiments
-------------------------------------

The official HypEx documentation can be found at:

`HypEx Documentation <https://hypex.readthedocs.io/en/latest/>`_

For a detailed reference, visit the HypEx API documentation.

Tabular interpretation with SSWARM
----------------------------------

SSWARM estimates Shapley values for fitted tabular models using sampled feature
coalitions and column permutations. It uses the input observations as the
background population for these permutations. Regression returns an array of
shape ``(samples, features)``; classification returns
``(classes, samples, features)``. Features follow ``explainer.used_feats``.
``explainer.expected_value`` contains the mean prediction on the input data.

.. code-block:: python

    from lightautoml.addons.tabular_interpretation import SSWARM

    explainer = SSWARM(automl, random_state=77)
    values = explainer.shap_values(
        test_data,
        T=None,
        n_repeats=3,
        batch_size=2 * len(test_data),
        n_jobs=1,
    )

Controlling memory
~~~~~~~~~~~~~~~~~~

``batch_size`` limits the number of rows in each prediction call. Each coalition
uses all input observations, so this limit must be at least ``len(test_data)``.
Smaller batches reduce prediction memory without changing the sampled coalitions
or column permutations for a fixed seed. The final batch contains only actual
observations; no padding rows are sent to the model. Prediction implementations
can still introduce their own numerical differences between batch sizes.

Only the reader's used features are passed to prediction. Numeric arrays keep
their NumPy dtype instead of being converted into arrays of Python objects.
Mixed feature types can still require object arrays. The input is not modified.

For ``m`` samples, ``n`` features and ``c`` outputs, the float64 result and
working accumulator occupy ``16 * m * n * c`` bytes together. Two integer count
matrices add ``16 * n ** 2`` bytes. This replaces four float64 arrays occupying
``32 * m * n ** 2 * c`` bytes, and avoids their four-dimensional subtraction
temporary. The coalition schedule, input data, prediction batches and model
workspace require additional memory. Working arrays and the coalition schedule
are released after a call, including when prediction raises an exception.

The schedule is retained for two passes: first count the contributions to each
feature/coalition-size stratum, then add predictions with those fixed
denominators directly to the accumulator. This computes the same sum of stratum
means, up to floating-point summation order, without storing every mean for
every observation and output.

``n_jobs > 1`` can increase total memory because prediction workers may hold
their own copies of the model and data. Use ``n_jobs=1`` when memory is limited;
it is required for models containing neural networks.

Sampling budget
~~~~~~~~~~~~~~~

``T`` is the total number of coalition evaluations per repeat, including exact
boundary coalitions and warm-up updates. For more than three features, the
minimum is ``2*n + 1 + 2*sum(ceil(n/s) for s in range(2, n-1))``. For example,
50 features require at least 497 evaluations and 100 features require 1,141.

``T=None`` chooses the larger of that minimum and the usual default of 250
evaluations below 20 features or 500 otherwise. An explicit smaller ``T`` raises
``ValueError`` before prediction. Earlier versions silently truncated the
initialization, leaving some strata unobserved and making results depend on
batch boundaries. Wide inputs can therefore take longer with the corrected
default. With one to three features, all nonempty coalitions are evaluated once
per repeat, even when a larger budget is requested; column permutations still
use the random seed. ``n_repeats`` averages estimates from successive seeds.

Reproducing a memory measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux or macOS, run each configuration in a fresh process from the repository
root. The benchmark uses a deterministic binary predictor to isolate explainer
memory; it does not measure a trained AutoML model's workspace.

.. code-block:: console

    PYTHONPATH=. python scripts/benchmark_sswarm_memory.py --batch-size 500000
    PYTHONPATH=. python scripts/benchmark_sswarm_memory.py --batch-size 8192

To compare revisions, run the same script with ``PYTHONPATH`` pointing to each
checkout. It reports peak process RSS, retained array storage, prediction row
count and elapsed time. RSS includes Python and imported libraries, and depends
on the platform and allocator; the figures are not a fixed memory guarantee.
