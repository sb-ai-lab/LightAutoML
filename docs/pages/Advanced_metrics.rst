Evaluation metrics
==================

Training and model selection
----------------------------

Choose a metric independently of the training loss. For example, optimize
logistic loss while selecting binary classifiers by average precision::

    from lightautoml.tasks import Task
    from lightautoml.automl.presets.tabular_presets import TabularAutoML

    task = Task("binary", metric="pr_auc")
    automl = TabularAutoML(task=task, timeout=60)
    predictions = automl.fit_predict(train_data, roles={"target": "target"})

Pass a dictionary for metric parameters. String expressions are not evaluated::

    task = Task("binary", metric="precision@k", metric_params={"k": 0.1})
    task = Task("binary", metric="fbeta_score",
                metric_params={"beta": 0.5, "cutoff": 0.7})
    task = Task("reg", metric="r2_adj", metric_params={"n_features": 12})
    task = Task("multilabel", metric="hamming_accuracy")

Named metrics carry their optimization direction. You do not need to infer it
from artificial predictions. ``Task.metric_func`` returns the raw value;
``Task.get_dataset_metric()`` negates losses so that model selection always
maximizes the returned score. An explicit ``greater_is_better`` overrides the
registered direction.

LightGBM and XGBoost use the new metrics through evaluation callbacks, with
their parameters and observation weights. LightGBM disables its default
evaluation metric when a Python metric is selected. Existing special-loss
overrides remain in place, including transformed-target losses.

CatBoost retains LightAutoML's existing callable-metric policy: metrics without
a native mapping use the training objective for early stopping. The requested
metric is still used by LightAutoML to compare, tune and blend models on
held-out predictions. This distinction applies to CPU and GPU training and
also to multioutput tasks. It does not make CatBoost early stopping optimize
AP@K, ECE or another unsupported metric. Existing LightGBM/XGBoost multioutput
restrictions are unchanged.

Standalone evaluation
---------------------

The new catalog can also be used without fitting a model::

    from lightautoml.tasks.metrics import get_metric

    metric = get_metric("pr_auc", task="binary")
    score = metric([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert metric.greater_is_better

    loss = get_metric("SMAPE", task="reg")
    score = loss([0, 2, 4], [0, 1, 3], sample_weight=[1, 2, 1])
    assert not loss.greater_is_better

Inputs are numeric arrays, lists, Series or DataFrames. Classification uses
the label encoding of the LightAutoML reader: binary labels are 0/1;
multiclass labels index probability columns. Labels and predictions are never
independently re-encoded. Pandas inputs are consumed in row order, without
implicit index joins. Align them before calling a metric. Application-specific
CSV parsing, baseline/prod-score report tables and treatment-label conversion
remain the caller's responsibility.

All new Task metrics accept ``sample_weight``. Weights apply to rows; regression
outputs are averaged uniformly after computing each output's metric. Custom
metrics reject negative/nonfinite weights and zero total weight. Standard
scikit-learn functions keep their upstream validation and restrictions;
in particular, multiclass OVO AUC does not support observation weights.

Classification catalog
----------------------

All names below are accepted by ``Task`` for the indicated task and by
``get_metric``. Existing names such as ``auc``, ``accuracy`` and
``crossentropy`` remain available through ``Task``.

.. list-table::
   :header-rows: 1
   :widths: 35 22 43

   * - Names
     - Tasks
     - Definition / parameters
   * - ``accuracy_score``
     - binary, multiclass, multilabel
     - Label accuracy; multilabel requires the whole row to match.
   * - ``f1_score``, ``precision_score``, ``recall_score``;
       short aliases ``f1``, ``precision``, ``recall``
     - binary
     - Positive-class scores; ``cutoff=0.5``, ``zero_division=0``.
   * - ``f1_macro``, ``f1_micro``, ``f1_weighted``;
       corresponding ``precision_*``, ``recall_*``, ``jaccard_*``
     - binary, multiclass, multilabel
     - scikit-learn averaging; ``zero_division=0`` by default.
   * - ``f1_average_samples``, ``precision_average_samples``,
       ``recall_average_samples``, ``jaccard_average_samples``;
       aliases ending in ``_samples``
     - multilabel
     - Compute each row's score, then average rows.
   * - ``fbeta_score``, ``fbeta``
     - binary, multiclass, multilabel
     - Required ``beta`` may be fractional; binary or macro averaging.
   * - ``roc_auc_score``, ``gini``
     - binary
     - ROC AUC; Gini = 2 * AUC - 1, in [-1, 1].
   * - ``pr_auc``, ``average_precision``
     - binary, multilabel
     - scikit-learn average precision, not trapezoidal PR area;
       multilabel uses macro averaging.
   * - ``logloss``
     - binary, multiclass
     - Probability log loss; multiclass alias for cross-entropy.
   * - ``brier_loss``, ``brier_score``
     - binary, multilabel
     - Mean squared probability error; multilabel averages independent labels.
   * - ``ece``
     - binary, multiclass, multilabel
     - L1 calibration error; ``n_bins=10`` equal-width bins.
   * - ``rec@spec``, ``recall_at_specificity``
     - binary
     - Maximum recall at specificity >= ``threshold`` (default 0.99).
   * - ``aucroc@fpr``, ``partial_auc``
     - binary
     - scikit-learn standardized partial ROC AUC up to FPR ``threshold``
       (default 0.99).
   * - ``roc_auc_score_ovr``, ``roc_auc_score_ovo``
     - multiclass
     - Macro one-vs-rest / one-vs-one ROC AUC.
   * - ``roc_auc_score_micro``, ``roc_auc_score_macro``,
       ``roc_auc_score_samples``
     - multilabel
     - scikit-learn ROC AUC with the specified averaging.
   * - ``roc_auc_with_nans``, ``roc_auc_score_average_samples``
     - multilabel
     - Macro average over valid columns, skipping missing targets and
       columns without both classes. The second name preserves the work alias;
       despite that name it is not sample-wise averaging.
   * - ``gini_with_nans``
     - multilabel
     - NaN-aware macro Gini, in [-100, 100].
   * - ``hamming_accuracy``
     - multilabel
     - One minus Hamming loss; a perfect prediction scores 1.

All classification metrics above are maximized except log loss, Brier loss
and ECE. Label metrics use ``prediction >= cutoff`` for binary/multilabel
and ``argmax`` for multiclass. Probability metrics never threshold their inputs.
The legacy binary ``accuracy`` name retains its existing strict ``> 0.5`` rule;
``accuracy_score`` uses the configurable inclusive cutoff.

ECE uses positive-class probability for binary classification, top-label
confidence/correctness for multiclass, and the mean of binary ECEs for
multilabel. Bins are right-closed and include both endpoints: zero belongs to
the first bin and one to the last. ECE is in [0, 1]. The implementation uses
NumPy bin accumulation, following the L1 definition documented by
`TorchMetrics <https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error>`_.

NaN-aware AUC accepts missing targets only. It raises ``ValueError`` if no
column has both classes with positive weight. This evaluates incomplete
labels; it does not add support for training models with missing labels.

Regression catalog
------------------

The following apply to ``reg`` and ``multi:reg``. The work spellings (including
uppercase names) are retained.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Names
     - Definition
   * - ``r2_score``
     - scikit-learn R2, maximized.
   * - ``r2_adj``, ``adjusted_r2``
     - Adjusted R2; required nonnegative integer ``n_features``. Uses the
       count of positive-weight rows and requires n > n_features + 1.
   * - ``MAE``, ``mean_absolute_error``; ``MSE``, ``mean_squared_error``
     - Standard scikit-learn errors, minimized.
   * - ``RMSE``, ``rmse``
     - Mean of per-output square roots of MSE, as in scikit-learn RMSE.
   * - ``MAPE``
     - Relative MAPE, minimized; denominator max(abs(target), float64 epsilon).
       No multiplication by 100. The existing lowercase ``Task`` metric
       ``mape`` retains its earlier behavior.
   * - ``median_absolute_error``
     - scikit-learn median absolute error.
   * - ``median_absolute_percentage_error``
     - Median percentage error, multiplied by 100; epsilon denominator.
       Weighted evaluation uses the lower weighted median of each output.
   * - ``mean_squared_log_error``, ``msle``
     - scikit-learn MSLE; requires nonnegative targets and predictions.
   * - ``SMAPE``, ``smape``
     - 200 * mean(abs(error) / (abs(target) + abs(prediction))).
       A zero/zero pair contributes zero; range [0, 200].
   * - ``1-WAPE``, ``one_minus_wape``
     - One minus MAE / mean(abs(target)), maximized.
   * - ``wape``, ``mae_to_module_mean``
     - MAE / mean(abs(target)), minimized, relative scale.
   * - ``mae_to_mean``
     - MAE / mean(target), minimized; every output must have a positive mean.
   * - ``mape_without_zeros``
     - Relative MAPE excluding zero targets per output with the same mask
       applied to predictions and weights.
   * - ``spearman``
     - Mean correlation of average ranks, maximized. With weights, compute
       weighted Pearson correlation of ranks after dropping zero-weight rows.
   * - ``powerstat``
     - Normalized Gini/Lorenz area * 100, maximized; nonnegative targets.

Relative MAE/WAPE with a zero denominator and MAPE without any nonzero target
raise ``ValueError``. They do not return a misleading negative error sentinel.
Powerstat requires nonconstant targets; equal prediction scores are grouped
before integrating the Lorenz curve, making the result independent of their
input order. Spearman raises for constant targets or predictions.

Ranking and both AP@K conventions
---------------------------------

For binary classification, ``precision@k`` and ``recall@k`` rank observations
by their scores. An integer ``k`` is a count capped at sample size; a float
in (0, 1] is a fraction, rounded down but selecting at least one observation.
Zero-weight rows do not occupy ranks. Ties preserve input order. Empty input
and nonpositive/invalid k raise ``ValueError``. No relevant observations gives
recall/AP zero.

``ap@k``, ``average_precision@k`` and ``average_precision_at_k`` compute
truncated average precision with the total number (or weight) of relevant
observations in the full sample as denominator. Relevant observations below
k remain in the denominator. ``average_precision_in_top_k`` preserves the
work formula: select top k, then call scikit-learn average precision only on
that subset. The latter also uses scikit-learn's score-tie averaging::

    from lightautoml.tasks.metrics import get_metric

    y = [1, 0, 1, 1]
    p = [0.9, 0.8, 0.7, 0.6]
    ap = get_metric("ap@k", k=2)(y, p)  # 1/3
    subset_ap = get_metric("average_precision_in_top_k", k=2)(y, p)  # 1

For query-item matrices, use ``Task("multilabel", metric=...)`` when the
underlying prediction problem is multilabel, or standalone
``get_metric(..., task="ranking")``. Weights then apply to queries (rows).
This adds evaluation metrics, not a group-aware ranking training task.

* ``map@k`` / ``mean_average_precision_at_k``: mean truncated AP@K.
* ``mean_average_precision_in_top_k``: mean work-formula subset AP.
* ``ndcg_score`` / ``ndcg@k`` / ``ndcg``: scikit-learn NDCG, with nonnegative
  relevance, linear gain and score-tie averaging; optional ``k``.
* ``label_ranking_average_precision_score``: scikit-learn LRAP, maximized.
* ``label_ranking_loss`` and ``coverage_error``: scikit-learn metrics, minimized.
* ``mrr`` for ``multilabel``: reciprocal rank of the first relevant item in
  each score row; optional ``k`` and stable input-order ties.

The work MRR takes item IDs instead of scores. It is available as
``get_metric("mrr", task="ranking")`` or ``mean_reciprocal_rank`` in
``lightautoml.tasks.metrics.ranking``::

    score = get_metric("mrr", task="ranking")(
        [["item-b"], ["item-x"]],
        [["item-a", "item-b"], ["item-y", "item-z"]],
    )  # 0.25

For a relevance/score matrix in standalone ranking mode use
``get_metric("reciprocal_rank_score", task="ranking")``.

Uplift and rolling forecasts
----------------------------

These helpers use explicit evaluation inputs, rather than new ``Task`` types::

    from lightautoml.tasks.metrics import get_metric

    metric = get_metric("uplift@k", task="uplift", k=0.5)
    score = metric([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0])

    metric = get_metric("mean_mae", task="timeseries", start_indices=[2, 3])
    score = metric([1, 2, 3, 4, 5], [[3, 4], [4, 6]])  # 0.25

Uplift names are ``qini_auc_score`` (``negative_effect=True`` by default),
``uplift_auc_score``, ``uplift@k`` / ``uplift_at_k`` (``k=0.5``), and
``weighted_average_uplift`` (``bins=10``). All are maximized. Treatment must
contain both groups, explicitly encoded as 0/1; outcomes are binary.
Observation weights are not supported by these uplift helpers.

Formulas, normalization and tie conventions follow
`scikit-uplift 0.5.1 <https://www.uplift-modeling.com/en/latest/api/metrics/index.html>`_,
adapted under its MIT license using NumPy without an added dependency.
Normalized AUC subtracts the random-policy baseline and divides by the
corresponding perfect-curve area above that baseline. These scores are not
aliases for LightAutoML's existing addon ``calculate_uplift_auc`` modes.

Top-k uplift and weighted average uplift accept ``strategy="overall"`` or
``"by_group"``. Their k restrictions match scikit-uplift: integer k must be
less than sample size; a fraction must be strictly between zero and one.
Both groups must be present in each evaluated top selection/bin. A zero
normalizing area raises ``ValueError``. Qini with ``negative_effect=False``
requires positive overall uplift.

``mean_mae`` and ``mean_mse`` average complete forecast-horizon errors over
forecast origins. By default forecast row i starts at observed series position
i. ``start_indices`` allows a compact matrix of forecasts beginning later in
the series. Leading scalar NaN placeholders (the work format) are accepted;
later missing forecasts, partial NaN horizons and unequal horizon lengths are
errors. Trailing horizons extending past the observed series are skipped.
Weights apply to forecast origins. At least one complete positive-weight
horizon is required. Ordinary aligned forecast matrices can instead use
``Task("multi:reg", metric="MAE")`` or ``"MSE"``.

Implementation references
-------------------------

Standard scores delegate to `scikit-learn metrics
<https://scikit-learn.org/stable/api/sklearn.metrics.html>`_. In particular,
``pr_auc`` uses `average precision
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`_,
and NDCG uses the documented `scikit-learn tie handling
<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html>`_.
Spearman ranking uses SciPy average ranks. Custom NumPy implementations cover
the work-specific formulas, calibration bins and weighted reductions.
