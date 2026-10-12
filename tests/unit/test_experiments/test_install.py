"""Regression tests for installing benchmarks in a Docker virtual environment."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
INSTALLER = REPOSITORY / "scripts" / "experiments" / "utils.py"
RUN_INSTALLER = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("benchmark_utils", sys.argv[1])
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
utils.install_lightautoml()
"""
FAKE_PIP = """
import json
import os
import sys
from pathlib import Path

with open(os.environ["BENCHMARK_PIP_LOG"], "a") as output:
    output.write(json.dumps({
        "python": sys.executable,
        "args": sys.argv[1:],
        "cwd": str(Path.cwd()),
    }) + "\\n")
sys.exit(int(os.environ.get("BENCHMARK_PIP_EXIT", "0")))
"""


class TestBenchmarkInstallation(unittest.TestCase):
    """Exercise real subprocesses without downloading ML dependencies."""

    def setUp(self):
        """Create a repository without a relative bin/pip or commands on PATH."""
        temporary = tempfile.TemporaryDirectory(prefix="benchmark install ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(REPOSITORY / "scripts" / "poetry_fix.py", self.root / "scripts" / "poetry_fix.py")
        (self.root / "pyproject.toml").write_text('[tool.poetry.dependencies]\npython = "~3.10.0"\n')
        (self.root / "pip.py").write_text(FAKE_PIP)
        self.log = self.root / "pip_calls.jsonl"
        self.environment = dict(
            os.environ,
            PATH="",
            PYTHONPATH=str(self.root),
            BENCHMARK_PIP_LOG=str(self.log),
            BENCHMARK_PIP_EXIT="0",
        )

    def run_installer(self):
        """Run the installer with the active interpreter in the fake repository."""
        return subprocess.run(
            [sys.executable, "-c", RUN_INSTALLER, str(INSTALLER)],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_install_uses_current_python_without_relative_bin_or_path(self):
        """Install the checkout with the current Python, without downgrading packaging."""
        result = self.run_installer()

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["python"], sys.executable)
        self.assertEqual(calls[0]["args"], ["install", "."])
        self.assertEqual(calls[0]["cwd"], str(self.root))
        self.assertIn('python = ">=3.10,<4.0"', (self.root / "pyproject.toml").read_text())

    def test_pip_failure_stops_the_installer(self):
        """Report installation failure instead of continuing to benchmark imports."""
        self.environment["BENCHMARK_PIP_EXIT"] = "7"

        result = self.run_installer()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CalledProcessError", result.stderr)
        self.assertIn("exit status 7", result.stderr)

    def test_preparation_failure_prevents_pip_installation(self):
        """Stop before pip when the Python compatibility preparation fails."""
        (self.root / "scripts" / "poetry_fix.py").unlink()

        result = self.run_installer()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CalledProcessError", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
