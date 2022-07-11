# Usefull scripts

1. ```poetry_fix.py``` fixes problem with too long time of poetry lock command. It installs a single python version in the ```pyproject.toml```, which helps resolve all dependencies in a short time.

    ```bash

    poetry run python poetry_fix.py [PYTHON_VERSION]  # Set single python version := {6, 7, 8, 9}

    poetry run python poetry_fix.py  # Set default version - all necessary pythons for library

    ```

**Warning**: You must set the default version before publishing the library to PyPI.
