from clearml import Dataset, Task
from tqdm import tqdm

names_versions = [[i["name"], i["version"]]
                  for i in Dataset.list_datasets(dataset_project="Datasets_with_metadata")]


for dataset_name, dataset_version in tqdm(names_versions):
    task = Task.create(project_name="Experiments_selection",
            task_name="task_{0}_{1}".format(dataset_name, dataset_version),
            script="experiments/run_selection_methods.py",
            packages=["pandas<2.0.0"],
            docker="python:3.8-bullseye",
            argparse_args=[("dataset", dataset_name),
                           ("dataset_version", dataset_version)])
    Task.enqueue(task, queue_name="test_cpu_queue")