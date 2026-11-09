"""Measure SSWARM peak RSS with a small deterministic predictor on Linux/macOS.

Run each configuration in a fresh process. Set PYTHONPATH to the checkout being
measured so the same script can compare revisions without copying model code.
"""

import argparse
import gc
import json
import resource
import sys
import time

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lightautoml.addons.tabular_interpretation import SSWARM


class SmallModel:
    """Isolate explainer allocations from training and large model workspaces."""

    def __init__(self, n_features):
        self.reader = SimpleNamespace(
            used_features=[f"f{i}" for i in range(n_features)],
            _n_classes=2,
            task=SimpleNamespace(name="binary"),
        )
        self.prediction_rows = 0

    def predict(self, data, features_names=None, n_jobs=1):
        """Return deterministic binary probabilities using two input columns."""
        self.prediction_rows += len(data)
        first = np.asarray(data[:, 0], dtype=np.float64)
        second = np.asarray(data[:, 1], dtype=np.float64)
        values = 1 / (1 + np.exp(-(first + 2 * second)))
        return SimpleNamespace(data=values[:, None])


def main():
    """Report peak process memory, retained array storage and elapsed time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--features", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.features < 4:
        parser.error("use at least four features when comparing with the original implementation")
    data = np.random.default_rng(123).normal(size=(args.rows, args.features)).astype(np.float32)
    model = SmallModel(args.features)
    explainer = SSWARM(model, random_state=77)
    gc.collect()
    started = time.perf_counter()
    values = explainer.shap_values(
        data,
        feature_names=model.reader.used_features,
        T=args.iterations,
        n_repeats=args.repeats,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started
    retained = sum(value.nbytes for value in vars(explainer).values() if isinstance(value, np.ndarray))
    # ru_maxrss is in bytes on macOS and KiB on Linux.
    rss_unit = 1 if sys.platform == "darwin" else 1024
    print(
        json.dumps(
            dict(
                vars(args),
                python=sys.version.split()[0],
                numpy=np.__version__,
                pandas=pd.__version__,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit / 2 ** 20,
                retained_array_mib=retained / 2 ** 20,
                result_mib=values.nbytes / 2 ** 20,
                prediction_rows=model.prediction_rows,
                seconds=elapsed,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
