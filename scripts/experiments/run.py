"""Run tabular automl using ClearML logging."""
import argparse
import os
import pandas as pd
import clearml
import numpy as np


def bench_bonus(path_to_bonus, task_name, project):  # noqa D103
    from run_tabular import main as run_tabular

    for inner_dir in os.listdir(path_to_bonus):
        if not os.path.isdir(f"{path_to_bonus}/{inner_dir}"):
            continue

        for dirname in os.listdir(f"{path_to_bonus}/{inner_dir}"):
            if dirname.startswith("."):
                continue

            full_dir = f"{path_to_bonus}/{inner_dir}/{dirname}"
            if not os.path.isdir(full_dir):
                continue
            print(full_dir)

            train_name = None
            test_name = None
            for fname in os.listdir(full_dir):
                if "train" in fname:
                    train_name = fname
                elif ("test" in fname) or ("OOT" in fname):
                    test_name = fname
            assert train_name is not None and test_name is not None

            train, test = pd.read_csv(os.path.join(full_dir, train_name)), pd.read_csv(
                os.path.join(full_dir, test_name)
            )
            # Lower column names in train and test dataframes
            train.columns = [col.lower() for col in train.columns]
            test.columns = [col.lower() for col in test.columns]

            if len(train["target"].unique()) == 2:
                task_type = "binary"
            elif (len(train["target"].unique()) > 2) and (len(train["target"].unique()) < 100):
                task_type = "multiclass"
            else:
                task_type = "regression"

            try:
                run_tabular(
                    project=project,
                    train=train,
                    test=test,
                    task_type=task_type,
                    task_name=task_name,
                    dataset_name=None,
                    cpu_limit=16,
                    memory_limit=32,
                    save_model=False,
                    dataset_id=os.path.join(inner_dir, dirname),
                )
            except Exception as e:
                print(f"Error processing {full_dir}: {e}")
                continue

    return


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
    parser.add_argument("--bonus", type=str, help="path to bonus", default=None)
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

    if args.bonus is not None:
        bench_bonus(args.bonus, task_name=args.name, project=args.project)
    else:
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
