#!/bin/bash

set -e
PACKAGE_NAME=LightAutoML_LAMA
source ./lama_venv/bin/activate

# install dev dependencies
poetry install

# Run demos

cd tests
pytest demo*
cd ..
