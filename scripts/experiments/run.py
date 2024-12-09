"""Run tabular automl using ClearML logging."""
import argparse
import os
import pandas as pd
import clearml


def main(  # noqa D103
    dataset_name: str,
    queue: str,
    image: str,
    project: str,
    cpu_limit: int,
    memory_limit: int,
    tags: list,
    dataset_project: str = None,
    dataset_partial_name: str = None,
):

    if (dataset_project is not None) or (dataset_partial_name is not None) or len(tags) > 0:
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

    else:
        dataset_list = [dataset_name]

    print(f"Running {len(dataset_list)} datasets:")

    for dataset in dataset_list[:3]:
        if isinstance(dataset, str):
            dataset_name = dataset
            tags = ""
        else:
            dataset_name = dataset["name"]
            tags = f"--tags {' '.join(dataset['tags'])}" if len(tags) else ""

        os.system(
            f'clearml-task --project {project} --name {dataset_name} --script scripts/experiments/run_tabular.py --queue {queue} {tags} --docker {image} --docker_args "--cpus={cpu_limit} --memory={memory_limit}g" --args dataset={dataset_name}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="dataset name or id", default="sampled_app_train")
    parser.add_argument("--dataset_project", type=str, help="dataset_project", default=None)
    parser.add_argument("--dataset_partial_name", type=str, help="dataset_partial_name", default=None)
    parser.add_argument("--tags", nargs="+", default=[], help="tags")
    parser.add_argument("--cpu_limit", type=int, help="cpu limit in n threads", default=8)
    parser.add_argument("--memory_limit", type=int, help="mem limit in GBs", default=16)
    parser.add_argument("--queue", type=str, help="clearml workers queue", default="cpu_queue")
    parser.add_argument("--project", type=str, help="clearml project", default="junk")
    parser.add_argument("--image", type=str, help="docker image", default="for_clearml:latest")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        cpu_limit=args.cpu_limit,
        memory_limit=args.memory_limit,
        dataset_partial_name=args.dataset_partial_name,
        dataset_project=args.dataset_project,
        tags=args.tags,
        queue=args.queue,
        project=args.project,
        image=args.image,
    )
