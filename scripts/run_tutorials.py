"""Tutorials runner."""

# -- Information --
#  Command for running jupyter-notebook from cli:
#    jupyter nbconvert --config .jupyter/jupyter_notebook_config.py --to notebook --execute --inplace $FILE_PATH


import subprocess

from pathlib import Path
from typing import List
from typing import Tuple

import click


DEFAULT_EXECLUDE_TUTORIALS = (2, 4)  # 2 -> spark, 4 - nlp(+gpu)
TUTORIALS_DIR = Path("examples/tutorials/")
TUTORIAL_PREFIX = "Tutorial"
JUPYTER_NBCONVERT_CMD_FMT = "jupyter nbconvert --config {CONFIG} --to notebook --execute --inplace {FILE}"

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def _starts_with_any(s: str, prefixs: Tuple[int]) -> bool:
    cond = (s.startswith("{}_{}".format(TUTORIAL_PREFIX, p)) for p in prefixs)
    return any(cond)


def get_valid_tutorials_paths(exe_tutorials: Tuple[int], exclude_tutorials: Tuple[int]) -> List[Path]:
    """List of tutorials for executions.

    Tutorial selection rule:
        1) Starts with `TUTORIAL_PREFIX`.
        2) If `exe_tutorials` isn't empty, append tutorial only from list `exe_tutorials`.
        3) Exclude tutorial with index from list `exclude_tutorials`.

    Args:
        exe_tutorials: List of tutorials number.
        exclude_tutorials: List of tutorials which should be excluded from run.

    Returns:
        List of tutorials' path.

    """
    tutorials = []
    for file_path in sorted(TUTORIALS_DIR.iterdir()):
        file_name = file_path.name

        if not file_name.startswith(TUTORIAL_PREFIX):
            continue

        if exe_tutorials and _starts_with_any(file_name, exe_tutorials):
            tutorials.append(file_path)
            continue

        if exclude_tutorials and _starts_with_any(file_name, exclude_tutorials):
            continue

        if not exe_tutorials:
            tutorials.append(file_path)

    return tutorials


def clean_tutorials_dir():
    """Clean generated files after runs."""

    def _rm_tree(pth):
        pth = Path(pth)
        if pth.is_dir():
            for child in pth.glob("*"):
                if child.is_file():
                    child.unlink()
                else:
                    _rm_tree(child)
            pth.rmdir()
        else:
            pth.unlink()

    for file_path in TUTORIALS_DIR.iterdir():
        if file_path.name.startswith(TUTORIAL_PREFIX):
            continue

        _rm_tree(file_path)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--config", default=".jupyter/jupyter_notebook_config.py", help="Path to jupyter config.")
@click.option(
    "--tutorial",
    "-t",
    multiple=True,
    type=int,
    help="Indexes of tutorial for execution. If absent will be run all tutorials.",
)
@click.option(
    "--exclude-tutorials",
    "-e",
    multiple=True,
    type=int,
    help="Indexes of the tutorials that should be excluded.",
)
@click.option("--ignore-exclude", "-i", is_flag=True, help="Ignore default excluded tutorials.")
@click.option("--clean", "-c", is_flag=True, help="Clean directory before execution.")
def main(config: str, tutorial: Tuple[int], exclude_tutorials: Tuple[int], ignore_exclude: bool, clean: bool):
    """Run tutorials."""
    if ignore_exclude:
        exclude_tutorials = tuple()
    else:
        if not exclude_tutorials:
            exclude_tutorials = DEFAULT_EXECLUDE_TUTORIALS

    if clean:
        clean_tutorials_dir()

    if not Path(config).exists():
        raise RuntimeError("Config file doesn't exist.")

    for filename in get_valid_tutorials_paths(tutorial, exclude_tutorials):
        cmd = JUPYTER_NBCONVERT_CMD_FMT.format(CONFIG=config, FILE=filename)
        subprocess.run(cmd, shell=True, check=True)

    clean_tutorials_dir()


if __name__ == "__main__":
    main()
