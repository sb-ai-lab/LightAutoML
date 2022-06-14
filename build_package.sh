#!/bin/bash

python3 -m venv lama_venv
source ./lama_venv/bin/activate

pip install -U pip
pip install -U poetry

poetry lock
poetry install --no-dev
poetry build
