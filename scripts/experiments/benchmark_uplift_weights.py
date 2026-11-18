"""Compare X-learner weighting on synthetic data with known treatment effects.

Run from the repository root with PYTHONPATH=. This is a diagnostic experiment,
not a claim of general superiority or a comparison under equal tuning budgets.
"""

import argparse
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from lightautoml.addons.uplift.metalearners import XLearner
from lightautoml.addons.uplift.utils import create_linear_automl
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


def run(seed, treatment_rate):
    """Evaluate old and corrected weights using identical fitted components."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(4000, 5))
    treatment = rng.binomial(1, treatment_rate, len(x))
    effect = 1 + x[:, 0] + 0.5 * x[:, 1]
    baseline = 3 * np.sin(x[:, 0]) + 2 * x[:, 1] ** 2 + x[:, 2] * x[:, 3]
    target = baseline + treatment * effect + rng.normal(scale=0.5, size=len(x))
    data = pd.DataFrame(x, columns=list("abcde"))
    data["category"] = np.where(x[:, 4] > 0, "a", "b")
    data["w"] = treatment
    data["y"] = target

    outcome = TabularAutoML(
        task=Task("reg"),
        timeout=120,
        cpu_limit=1,
        gpu_ids="",
        general_params={"use_algos": [["cb"]]},
        reader_params={"cv": 3, "advanced_roles": False},
        selection_params={"mode": 0},
        cb_params={
            "default_params": {"num_trees": 120, "max_depth": 5, "learning_rate": 0.08},
            "freeze_defaults": True,
        },
    )

    def linear(task):
        learner = create_linear_automl(Task(task), n_folds=3, cpu_limit=1)
        learner.reader.advanced_roles = False
        return learner

    model = XLearner(
        outcome_learners=[outcome],
        effect_learners=[linear("reg")],
        propensity_learner=linear("binary"),
    )
    train = data.iloc[:2000]
    test = data.iloc[2000:].drop(columns=["y", "w"])
    model.fit(train, {"target": "y", "treatment": "w", "category": ["category"]})
    corrected, outcome_treatment, outcome_control = model.predict(test)
    propensity = model.learners["propensity"].predict(test).data.ravel()
    effect_control = model.learners["effect"]["control"].predict(test).data.ravel()
    effect_treatment = model.learners["effect"]["treatment"].predict(test).data.ravel()
    old = propensity * effect_treatment + (1 - propensity) * effect_control

    direct_predictions = []
    for group in [0, 1]:
        subset = train[train.w == group]
        direct = CatBoostRegressor(
            iterations=120,
            depth=5,
            learning_rate=0.08,
            thread_count=1,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        direct.fit(subset.drop(columns=["y", "w"]), subset.y, cat_features=["category"])
        direct_predictions.append(direct.predict(test))

    truth = effect[2000:]
    return {
        "seed": seed,
        "treatment_rate": treatment_rate,
        "x_old_mse": mean_squared_error(truth, old),
        "x_corrected_mse": mean_squared_error(truth, corrected),
        "t_lama_mse": mean_squared_error(truth, outcome_treatment - outcome_control),
        "t_direct_catboost_mse": mean_squared_error(truth, direct_predictions[1] - direct_predictions[0]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--treatment-rates", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    args = parser.parse_args()
    for rate in args.treatment_rates:
        for seed in args.seeds:
            print(json.dumps(run(seed, rate)), flush=True)
