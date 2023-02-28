"""Specify python verion."""

import argparse
import fileinput
import re
import sys

from pathlib import Path
from typing import Optional


PYPROJECT_TOML = Path("pyproject.toml")
ALL_PYTHON_DEPS = ">=3.6.1, <3.10"
PYTHON_DEPS = {6: "~3.6.1", 7: "~3.7.1", 8: "~3.8.0", 9: "~3.9.0"}
PYTHON_DEPS_PATTERN = '^python = ".*"$'


def _check_py_version():
    py_version = sys.version_info

    if py_version.major != 3:
        raise RuntimeError("Works only with python 3")

    if py_version.minor not in PYTHON_DEPS:
        raise RuntimeError(f"Works only with python 3.[{list(PYTHON_DEPS)}]")


def _set_version(py_version: Optional[int] = None):
    for line in fileinput.input(PYPROJECT_TOML.name, inplace=1):
        if re.search(PYTHON_DEPS_PATTERN, line):
            if py_version is None:
                version = ALL_PYTHON_DEPS
            else:
                version = PYTHON_DEPS[py_version]
            line = 'python = "{}"\n'.format(version)

        sys.stdout.write(line)


def main():
    """Cli."""
    _check_py_version()

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--current",
        action="store_true",
        help="Set curret python version in `pyproject.toml`",
    )
    group.add_argument("-f", "--full", action="store_true", help="Set all pythons versions in `pyproject.toml`")

    args = parser.parse_args()

    if args.current:
        _set_version(sys.version_info.minor)
    elif args.full:
        _set_version(None)


if __name__ == "__main__":
    main()
