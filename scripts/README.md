# Useful scripts

1. ```run_tutorials.py``` - execute tutorials in CLI. The execution drops in case of an any error. More information in `help`.

    ```bash

    # Run all tutorials except those excluded by default.
    poetry run python scripts/run_tutorials.py

    # Run tutorials (1, 2)
    poetry run python scripts/run_tutorials -t 1 -t 2

    ```

## Benchmark in GitHub Actions

The `benchmark` workflow (`.github/workflows/benchmark.yml`)
is started manually from **Actions → benchmark → Run workflow**.
It submits benchmark tasks for the selected revision; completion and quality
review are separate steps. Setup, inputs, and release comparison procedures
are documented in the team's
[experimenttracking runbook](https://tech-point.ailabtools.org/ai-lab-pmo/mltools/automl/experimenttracking/-/blob/master/docs/release-benchmark.md).
