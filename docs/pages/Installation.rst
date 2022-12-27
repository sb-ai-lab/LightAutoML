Installation Guide
==================


Basic
-----

You can install library `LightAutoML` from PyPI.

.. code-block:: bash

    pip install lightautoml


Development
-----------

You can also clone repository and install with poetry.
First, install `poetry <https://python-poetry.org/docs/#installation>`_.
Then,

.. code-block:: bash

    git clone git@github.com:AILab-MLTools/LightAutoML.git
    cd LightAutoML

    # Create virtual environment inside your project directory
    poetry config virtualenvs.in-project true

    # If you want to update dependecies, run the command:
    poetry lock

    # Installation
    poetry install
