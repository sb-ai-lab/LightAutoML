import logging
import os
import sys


_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    _logger.propagate = False

__all__ = [
    "automl",
    "dataset",
    "ml_algo",
    "pipelines",
    "image",
    "reader",
    "transformers",
    "validation",
    "text",
    "tasks",
    "utils",
    "addons",
    "report",
]

