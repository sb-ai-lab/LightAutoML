"""Tests for submitting benchmark tasks without a ClearML server."""

import importlib.util
import subprocess
import sys
import unittest

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch


RUN_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "experiments" / "run.py"
SPEC = importlib.util.spec_from_file_location("benchmark_run", RUN_SCRIPT)
benchmark_run = importlib.util.module_from_spec(SPEC)
original_clearml = sys.modules.get("clearml")
sys.modules["clearml"] = MagicMock()
try:
    SPEC.loader.exec_module(benchmark_run)
finally:
    if original_clearml is None:
        sys.modules.pop("clearml")
    else:
        sys.modules["clearml"] = original_clearml


class TestBenchmarkSubmission(unittest.TestCase):
    def setUp(self):
        self.options = dict(
            task_name="github-123-1",
            dataset_name="dataset with spaces; $(echo test)",
            queue="cpu_queue",
            image="for_clearml:3.12",
            project="Benchmark project",
            cpu_limit=4,
            memory_limit=12,
            min_num_obs=None,
            tags=[],
            dataset_project="Datasets_with_metadata",
        )

    def test_single_dataset_preserves_arguments_and_commit(self):
        self.options.update(repo="https://github.com/example/LightAutoML.git", branch="development", commit="abc123")
        with patch.object(benchmark_run.subprocess, "run") as submit:
            with patch.object(benchmark_run.clearml.Dataset, "list_datasets") as list_datasets:
                benchmark_run.main(**self.options)

        list_datasets.assert_not_called()
        submit.assert_called_once()
        command = submit.call_args.args[0]
        self.assertEqual(submit.call_args.kwargs, {"check": True})
        self.assertEqual(command[command.index("--project") + 1], "Benchmark project")
        self.assertIn("dataset=" + self.options["dataset_name"], command)
        self.assertIn("cpu_limit=4", command)
        self.assertIn("memory_limit=12", command)
        self.assertIn("save_model=False", command)
        self.assertIn("--docker_args=--cpus=4 --memory=12g", command)
        for key in ("repo", "branch", "commit"):
            self.assertEqual(command[command.index("--" + key) + 1], self.options[key])

    def test_submission_failure_is_propagated(self):
        with patch.object(
            benchmark_run.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "clearml-task")
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                benchmark_run.main(**self.options)

    def test_all_datasets_includes_last_dataset_and_latest_versions(self):
        self.options.update(dataset_name=None, n_datasets=-1)
        datasets = [
            {"name": "first", "version": "1", "tags": ["binary"]},
            {"name": "first", "version": "2", "tags": ["binary", "latest"]},
            {"name": "last", "version": "1", "tags": []},
        ]
        with patch.object(benchmark_run.clearml.Dataset, "list_datasets", return_value=datasets):
            with patch.object(benchmark_run.subprocess, "run") as submit:
                benchmark_run.main(**self.options)

        self.assertEqual(submit.call_count, 2)
        commands = [call.args[0] for call in submit.call_args_list]
        self.assertEqual(
            {next(arg for arg in cmd if arg.startswith("dataset=")) for cmd in commands},
            {"dataset=first", "dataset=last"},
        )
        first = next(cmd for cmd in commands if "dataset=first" in cmd)
        self.assertIn("latest", first)
        self.assertEqual(datasets[1]["tags"], ["binary", "latest"])

    def test_dataset_limit_and_model_saving(self):
        self.options.update(dataset_name=None, n_datasets=1, save_model=True)
        datasets = [{"name": name, "version": "1", "tags": None} for name in ("first", "last")]
        with patch.object(benchmark_run.clearml.Dataset, "list_datasets", return_value=datasets):
            with patch.object(benchmark_run.subprocess, "run") as submit:
                benchmark_run.main(**self.options)
        submit.assert_called_once()
        self.assertIn("save_model=True", submit.call_args.args[0])

    def test_empty_dataset_selection_fails_before_submission(self):
        self.options["dataset_name"] = None
        with patch.object(benchmark_run.clearml.Dataset, "list_datasets", return_value=[]):
            with patch.object(benchmark_run.subprocess, "run") as submit:
                with self.assertRaisesRegex(ValueError, "dataset was found"):
                    benchmark_run.main(**self.options)
        submit.assert_not_called()

    def test_invalid_limits_fail_before_submission(self):
        for key, value in (("n_datasets", 0), ("n_datasets", -2), ("cpu_limit", 0), ("memory_limit", -1)):
            with self.subTest(key=key, value=value):
                options = dict(self.options, **{key: value})
                with patch.object(benchmark_run.subprocess, "run") as submit:
                    with self.assertRaises(ValueError):
                        benchmark_run.main(**options)
                submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
