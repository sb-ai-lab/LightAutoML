import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from lightautoml.addons.autots.base import AutoTS
from lightautoml.tasks import Task


np.random.seed(42)

data = pd.read_csv("data/ai92_value_77.csv")
horizon = 30

train = data[:-horizon]
test = data[-horizon:]

roles = {"target": "value", "datetime": "date"}

seq_params = {
    "seq0": {
        "case": "next_values",
        "params": {"n_target": horizon, "history": np.maximum(7, horizon), "step": 1, "test_last": True},
    },
}

# True (then set default values) / False; int, list or np.array
# default: lag_features=30, diff_features=7
transformers_params = {
    "lag_features": [0, 1, 2, 3, 5, 10],
    "lag_time_features": [0, 1, 2],
    "diff_features": [0, 1, 3, 4],
}

task = Task("multi:reg", greater_is_better=False, metric="mae", loss="mae")

automl = AutoTS(
    task,
    seq_params=seq_params,
    trend_params={
        "trend": False,
    },
    transformers_params=transformers_params,
)
train_pred, _ = automl.fit_predict(train, roles, verbose=4)
forecast, _ = automl.predict(train)

print("Check scores...")
print("TEST score: {}".format(mean_absolute_error(test[roles["target"]].values, forecast.data)))
