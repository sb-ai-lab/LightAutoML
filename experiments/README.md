# How to run exmeriments with ClearML
For details check ExperimentTracking repo

### Set two remotes for a local repository
```
git remote -v
ssh://git@37.18.73.175:5109/ai-lab-pmo/mltools/automl/LightAutoML.git
```

### Run single experiment
```clearml-task --project PROJECT_NAME --name task --script experiments/run_tabular.py --queue QUEUE_NAME --requirements experiments/requirements.txt --docker for_clearml:latest --repo ```
