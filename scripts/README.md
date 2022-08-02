# Usefull scripts

1. ```poetry_fix.py``` fixes problem: too long time of poetry lock command. It re-writes a single python version in the ```pyproject.toml```, which helps resolve all dependencies in a short time.

    ```bash

    # Set single python version := {6, 7, 8, 9}
    # After that you can easily run command: `poetry lock`
    poetry run python poetry_fix.py [PYTHON_VERSION]

    # Set all default versions before `git push` or `poetry publish`
    poetry run python poetry_fix.py

    ```

**Warning**: You must set the default version before publishing the library to PyPI.
