"""Run tabular automl using ClearML logging."""
import argparse
import os
import pandas as pd
import clearml
import numpy as np


def main(  # noqa D103
    task_name: str,
    dataset_name: str,
    queue: str,
    image: str,
    project: str,
    cpu_limit: int,
    min_num_obs: int,
    memory_limit: int,
    tags: list,
    dataset_project: str = None,
    dataset_partial_name: str = None,
    n_datasets: int = -1,
    save_model: bool = False,
):
    if dataset_name is not None:
        dataset_list = [dataset_name]
    else:
        dataset_list = pd.DataFrame(
            clearml.Dataset.list_datasets(
                dataset_project=dataset_project,
                partial_name=dataset_partial_name,
                tags=tags,
                ids=None,
                only_completed=True,
                recursive_project_search=True,
                include_archived=False,
            )
        )
        dataset_list = (
            dataset_list.sort_values("version", ascending=False).drop_duplicates(subset=["name"]).to_dict("records")
        )

        if min_num_obs is not None:
            for indx, dataset in enumerate(dataset_list):
                metadata = clearml.Dataset.get(dataset_id=None, dataset_name=dataset["name"]).get_metadata()
                if metadata["num_obs"].iloc[0] < min_num_obs:
                    dataset_list.pop(indx)

        if len(dataset_list) <= 0:
            raise ValueError("No one dataset was found with passed parameters.")

        np.random.shuffle(dataset_list)
        dataset_list = dataset_list[:n_datasets]

    print(f"Running {len(dataset_list)} datasets:")

    for dataset in dataset_list:
        if isinstance(dataset, str):
            dataset_name = dataset
            tags = [""]
        else:
            dataset_name = dataset["name"]
            tags = dataset["tags"]

        curr_task_name = f"{task_name}@{dataset_name}" if task_name is not None else f"{dataset_name}"

        tags.append(queue)
        tags = f"--tags {' '.join(tags)}" if len(tags) else ""

        os.system(
            f'clearml-task --project {project} --name {curr_task_name} --script scripts/experiments/run_tabular.py --queue {queue} {tags} --docker {image} --docker_args "--cpus={cpu_limit} --memory={memory_limit}g" --args dataset={dataset_name} save_model={save_model}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name", type=str, help="name for task", default=None)
    parser.add_argument("--dataset", type=str, help="dataset name or id", default=None)
    parser.add_argument("--dataset_project", type=str, help="dataset_project", default="Datasets_with_metadata")
    parser.add_argument("--dataset_partial_name", type=str, help="dataset_partial_name", default=None)
    parser.add_argument("--tags", nargs="+", default=[], help="tags")
    parser.add_argument("--cpu_limit", type=int, help="cpu limit in n threads", default=8)
    parser.add_argument("--memory_limit", type=int, help="mem limit in GBs", default=16)
    parser.add_argument("--queue", type=str, help="clearml workers queue", default="cpu_queue")
    parser.add_argument("--project", type=str, help="clearml project", default="junk")
    parser.add_argument("--image", type=str, help="docker image", default="for_clearml:latest")
    parser.add_argument("--n_datasets", type=int, help="number of datasets", default=-1)
    parser.add_argument("--min_num_obs", type=int, help="min number of samples", default=None)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    main(
        task_name=args.name,
        dataset_name=args.dataset,
        cpu_limit=args.cpu_limit,
        memory_limit=args.memory_limit,
        dataset_partial_name=args.dataset_partial_name,
        dataset_project=args.dataset_project,
        tags=args.tags,
        queue=args.queue,
        project=args.project,
        image=args.image,
        n_datasets=args.n_datasets,
        min_num_obs=args.min_num_obs,
        save_model=args.save_model,
    )
