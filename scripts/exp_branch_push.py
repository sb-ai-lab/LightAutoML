"""Experiment branch filter."""

import os
import subprocess


LAMA_GITHUB_URL = "git@github.com:sb-ai-lab/LightAutoML.git"
EXPERIMENT_BRANCH_PREFIX = "experiment/"

REMOTE_URL = os.getenv("PRE_COMMIT_REMOTE_URL")
BRANCH_NAME = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE).stdout.decode(
    "utf-8"
)

if BRANCH_NAME.startswith(EXPERIMENT_BRANCH_PREFIX) and REMOTE_URL == LAMA_GITHUB_URL:
    raise RuntimeError("Prevent push 'experiment/' branches to LAMA Github")
