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
AFG is also tested with pandas 2 (``tests/constraints/afg.txt``) and the complete
``all`` extra with pandas 3 (``tests/constraints/afg-pandas3.txt``). Both jobs
check dependency consistency and exercise feature selection, relational and
temporal features, serialization and parallel computation.
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

The optional ``afg``/``all`` extras support pandas 2 and 3. Woodwork through 0.31
relies on cached accessors, removed in pandas 3. On pandas 3 only, constructing
or unpickling a ``FeatureGeneratorTransformer`` installs an adapter that caches
``DataFrame.ww`` and ``Series.ww`` on their owners. Importing LightAutoML or its
generator module does not activate it. Initialization is idempotent; pandas 2
and Woodwork 0.32 or later retain their original behavior. The version guard
should be revisited when Woodwork releases native pandas 3 support.

The replacement descriptors affect **all Woodwork users in the process**.
Other pandas accessors and pandas' general accessor implementation are unchanged.
Woodwork uses weak references to the owning objects, so caching does not add
a reference cycle. Dask workers initialize the adapter before receiving AFG's
EntitySets. Parallel AFG requires Dask's optional distributed dependencies
(``pip install "dask[distributed]"``), as required by Featuretools.
LightAutoML calls Featuretools' public serial calculator in Dask workers to
avoid the obsolete Dask APIs in Featuretools 1.31's parallel backend
(`upstream issue <https://github.com/alteryx/featuretools/issues/2767>`_).
Workers receive the complete EntitySet and a common cutoff time. Feature
columns are partitioned instead of rows, preserving history for order-dependent
primitives such as ``TIME_SINCE_PREVIOUS``. The number of workers is capped by
the number of requested features.

On pandas 2, Featuretools 1.31's use of NumPy ``sum`` in groupby emits a
``FutureWarning`` about pandas' future callable dispatch. The compatibility
tests exempt this specific upstream warning on pandas 2 only. Application
warning filters and aggregation semantics remain unchanged.

The ``afg``/``all`` extras still require ``setuptools<82`` because Woodwork
imports ``pkg_resources`` (removed in setuptools 82). Woodwork also emits a
deprecation warning for that import. This restriction does not apply to the
core, NLP, CV or report extras.
See the
`setuptools removal notice <https://setuptools.pypa.io/en/latest/pkg_resources.html>`_.

Saved neural models
-------------------

At prediction time, unavailable CUDA devices fall back to CPU. When a saved
CUDA index no longer exists but CUDA is available, the trainer uses the first
available GPU. Constructor parameters are updated before modules allocate
tensors. Tensors created during forward passes follow the input device.
An explicit CPU request takes precedence over saved multi-GPU IDs and clears
the saved parallel-device selection.

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
Nullable pandas and Arrow numeric columns retain their dtypes and existing
missing values during sanitization.
Invalid targets and weights still raise the underlying model's error.

The Linux CUDA wheels of XGBoost 3.3.0 and 3.4.1 were observed to raise
``cudaErrorNoDevice`` for multiclass training with ``device="cpu"`` when an
NVIDIA driver is installed but all GPUs are hidden with
``CUDA_VISIBLE_DEVICES=""``. This also reproduces in a direct ``xgboost.train``
call without LightAutoML. Hiding GPUs on a driver-equipped machine therefore
does not reproduce the driver-free CPU environment used by compatibility CI.

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
