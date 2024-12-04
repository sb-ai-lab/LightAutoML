"""Run tabular automl using ClearML logging."""
import argparse
import os
import clearml


def main(  # noqa D103
    dataset_name: str,
    queue: str,
    project: str,
    cpu_limit: int,
    memory_limit: int,
    dataset_project: str = None,
    dataset_partial_name: str = None,
    tags=None,
):

    if (dataset_project is not None) or (dataset_partial_name is not None) or (tags is not None):
        tags = tags if isinstance(tags, list) else [tags]

        dataset_list = clearml.Dataset.list_datasets(
            dataset_project=dataset_project,
            partial_name=dataset_partial_name,
            tags=tags,
            ids=None,
            only_completed=True,
            recursive_project_search=True,
            include_archived=False,
        )
        print(dataset_list[0])
        dataset_list = list(set([x["name"] for x in dataset_list]))

    else:
        dataset_list = [clearml.Dataset.get(dataset_id=None, dataset_name=dataset_name)]

    print(f"Running {len(dataset_list)} datasets...")
    print(dataset_list)

    for dataset in dataset_list:
        os.system(
            f'clearml-task --project {project} --name {dataset} --script scripts/experiments/run_tabular.py --queue {queue} --docker for_clearml:latest --docker_args "--cpus={cpu_limit} --memory={memory_limit}g" --args dataset={dataset}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="dataset name or id", default="sampled_app_train")
    parser.add_argument("--dataset_project", type=str, help="dataset_project", default=None)
    parser.add_argument("--dataset_partial_name", type=str, help="dataset_partial_name", default=None)
    parser.add_argument("--tags", type=str, help="tags", default=None)
    parser.add_argument("--cpu_limit", type=int, help="", default=8)
    parser.add_argument("--memory_limit", type=int, help="", default=16)
    parser.add_argument("--queue", type=str, help="", default="cpu_queue")
    parser.add_argument("--project", type=str, help="", default="junk")
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
    )
