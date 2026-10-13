Dependency and device compatibility
===================================

The development branch supports Python 3.10 through 3.14. Package installers
choose dependency versions compatible with the interpreter and platform.
The lower Python requirement differs from the older 0.4.2 master branch;
this compatibility update does not further raise it.

Dependency updates
------------------

XGBoost 3.x, statsmodels 0.15, Albumentations 2.x, OpenCV 5.x, WeasyPrint 70
and CFFI 2.x are now allowed alongside earlier supported releases. Dependencies
that already had open version ranges continue to resolve to current releases.
The ``all`` extra includes ``timm``, and ``scikit-image`` is explicitly declared
for the existing CV extras. ``poetry-core`` is a build dependency only.

Reproducible dependency snapshots are in ``tests/constraints/legacy.txt`` and
``tests/constraints/latest.txt``. The latest snapshot was checked against PyPI
on 2026-10-13. These constraints apply to tests, not to user installations.
The dedicated compatibility workflow runs the regression tests without model
downloads or a GPU, with both older and current dependency versions.
The AFG extra is tested separately with ``tests/constraints/afg.txt``; its
pandas constraint cannot be combined with the pandas 3 snapshot.
Legacy jobs pin Arrow to a NumPy 1 compatible release and allow only the
specific ``ImportWarning`` from
`PyTorch 2.2's circular einops import <https://github.com/arogozhnikov/einops/issues/315>`_.
Current-dependency jobs treat that warning as an error.

String and Arrow-backed columns are accepted as explicit categorical features.
Ordinal ranking uses object-backed strings to avoid Arrow's deprecated rank
API. Existing category-mapping and automatic role-selection semantics are
retained; changing them would also change neural-network inputs and training.
Color histograms accept both OpenCV 4's column vectors and OpenCV 5's flat
arrays, and include the full uint8 range, including white pixels (255).

The optional ``afg``/``all`` extra currently requires ``pandas>=2,<3`` and
``setuptools<82``. Featuretools' current Woodwork dependency relies on cached
DataFrame accessors (removed in pandas 3), and still imports ``pkg_resources``
(removed in setuptools 82). These restrictions do not apply to the core,
NLP, CV or report extras, which can use pandas 3 and current setuptools.
Woodwork also emits a deprecation warning for its ``pkg_resources`` import.
See the
`setuptools removal notice <https://setuptools.pypa.io/en/latest/pkg_resources.html>`_.

Saved neural models
-------------------

At prediction time, unavailable CUDA devices fall back to CPU. When a saved
CUDA index no longer exists but CUDA is available, the trainer uses the first
available GPU. Constructor parameters are updated before modules allocate
tensors. Tensors created during forward passes follow the input device.

The checkpoint reader continues to accept existing in-memory and file-based
states, including NumPy snapshot-ensemble metadata. Checkpoints, like joblib
models, must come from trusted sources: loading their Python state can execute
code. Compatibility of arbitrary pickled pipelines across dependency major
versions is not guaranteed by this change.

Numeric data and binning
------------------------

XGBoost treats positive/negative infinity and feature values outside the
float32 range as missing values during training, validation and prediction.
Input datasets are not modified. Sparse inputs keep their sparse structure,
including the distinction between explicit zero and a missing sparse entry.
Pandas inputs retain column names and indexes for XGBoost's feature validation.
Invalid targets and weights still raise the underlying model's error.

Only piecewise numeric embeddings compute quantile bins. For compatibility,
bins retain their existing fitting population: the first fold's combined
training and validation data. They are computed in float32 and cached in model
parameters, as in earlier releases. This preserves the existing checkpoint
layout and does not change cross-validation preprocessing semantics.

For small samples the number of bins is reduced automatically; constant
features receive a finite interval. Training batches are reduced when a fold
is smaller than the requested batch size, preventing an empty training loader.
Above 200,000 rows, quantiles use a
deterministic sample, processed one feature at a time, while retaining the
full input minimum and maximum. This bounds quantile workspace and avoids
PyTorch's 2**24-row limit. The ``compute_bins`` function exposes ``subsample``
and ``random_state`` controls. Ordinary, unsampled quantiles are unchanged.

Validation scope
----------------

CPU tests cover stale CUDA parameters, constructor device propagation,
checkpoint round trips, sequence embeddings, small and sampled large-input
binning, and XGBoost with dense and sparse non-finite features. Mocked CUDA
availability checks device selection; they do not validate CUDA kernels,
driver compatibility, real multi-GPU execution or GPU memory consumption.
Those checks require GPU hardware.
