# Useful scripts

1. ```poetry_fix.py``` fixes problem: too long time of poetry lock command. It re-writes a single python version in the ```pyproject.toml```, which helps resolve all dependencies in a short time.

    ```bash

    # Set current python version from poetry env
    # After that you can easily run command: `poetry lock`
    poetry run python poetry_fix.py -c

    # Set all python versions before `git push` or `poetry publish`
    poetry run python poetry_fix.py -f

    ```

    **Warning**: You must set the default version before publishing the library to PyPI.

2. ```run_tutorials.py``` - execute tutorials in CLI. The execution drops in case of an any error. More information in `help`.

    ```bash

    # Run all tutorials except those excluded by default.
    poetry run python scripts/run_tutorials.py

    # Run tutorials (1, 2)
    poetry run python scripts/run_tutorials -t 1 -t 2

    ```

## ClearML benchmark in GitHub Actions

The `ClearML benchmark` workflow (`.github/workflows/benchmark_clearml.yml`)
is started manually from **Actions → ClearML benchmark → Run workflow**.
Select the branch or tag to benchmark and supply a dataset name, or leave it
empty to select random datasets from `dataset_project`. `n_datasets` defaults
to 5; use -1 to submit all matching datasets.

Configure the following in **Settings → Secrets and variables → Actions**:

| Kind | Name | Value |
| --- | --- | --- |
| Variable | `CLEARML_API_HOST` | ClearML API server URL |
| Variable | `CLEARML_WEB_HOST` | ClearML Web UI URL |
| Variable | `CLEARML_FILES_HOST` | ClearML file server URL |
| Secret | `CLEARML_API_ACCESS_KEY` | ClearML access key |
| Secret | `CLEARML_API_SECRET_KEY` | ClearML secret key |

The GitHub runner must be able to reach the ClearML API. A ClearML Agent must
listen on the selected queue and have access to the repository, datasets, and
Docker image. The default `for_clearml:3.12` image must already be available on
the worker; the workflow does not build it.

Each task runs `scripts/experiments/run_tabular.py` at the workflow's exact
commit. Task names start with `github-<run-id>-<run-attempt>@`. CPU and memory
limits apply to both the Docker container and LightAutoML. Submission errors
fail the workflow. A successful workflow means tasks were enqueued; follow
training status and benchmark metrics through task links in the job log or
the ClearML Web UI. The workflow does not wait for training to finish.
