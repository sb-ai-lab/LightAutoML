# How to run exmeriments with ClearML
For details check ExperimentTracking repo

### Run single experiment
```clearml-task --project PROJECT_NAME --name task --script experiments/run_tabular.py --queue QUEUE_NAME --requirements experiments/requirements.txt --docker for_clearml:latest --repo ```