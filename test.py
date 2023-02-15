import gdown

url = "https://drive.google.com/uc?export=download&id=1VLGS1LnL1NU28tpU-Hb9B2ymHj6z5qgq"
output = "hillstrom.csv"
gdown.download(url, output, quiet=False)

import pandas as pd
uplift_df = pd.read_csv("hillstrom.csv")

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

task = Task('multilabel', metric = 'logloss') 
roles = {'target': ['visit', 'mens', 'newbie'] }
print(uplift_df.columns)

automl = TabularAutoML(
    task = task, 
    timeout = 9999999999,
    general_params = {'use_algos': ['linear_l2', 'linear_l2', 'linear_l2']},
    nn_params={
            "n_epochs": 5, "bs": 2048, "num_workers": 0, "path_to_save": None,
        })

oof = automl.fit_predict(uplift_df.fillna(0), roles = roles, verbose = 5)
