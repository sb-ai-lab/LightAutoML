import json
import os
import pickle
import uuid
from traceback import extract_stack, format_stack
from typing import Any, Union, Dict

LOG_DATA_DIR = "LOG_DATA_DIR"
LAMA_LIBRARY = "LAMA_LIBRARY"


def is_datalog_enabled():
    return LOG_DATA_DIR in os.environ and len(os.environ[LOG_DATA_DIR]) > 0


def log_data(name: str, data: Any) -> None:
    if not is_datalog_enabled():
        return

    if "spark" not in name and "lama" not in name:
        library = os.environ[LAMA_LIBRARY] if LAMA_LIBRARY in os.environ else "unknownlib"
        name = f"{name}_{library}"

    base_path = os.environ[LOG_DATA_DIR]
    os.makedirs(base_path, exist_ok=True)

    log_record = {"stacktrace": extract_stack(), "data": data}

    filepath = os.path.join(base_path, f"datalog_{name}.pickle")
    with open(filepath, "wb") as f:
        pickle.dump(log_record, f)


def log_metric(name: str, event_name: str, metric_name: str, value: Union[str, dict]):
    if not is_datalog_enabled():
        return

    base_path = os.environ[LOG_DATA_DIR]
    os.makedirs(base_path, exist_ok=True)

    log_record = {"event": event_name, "metric_name": metric_name, "value": value}

    filepath = os.path.join(base_path, f"metriclog_{name}.json")
    with open(filepath, "a") as f:
        json.dump(log_record, f)
        f.write(os.linesep)


def log_config(name: str, config: Dict[str, Any]):
    if not is_datalog_enabled():
        return

    base_path = os.environ[LOG_DATA_DIR]
    os.makedirs(base_path, exist_ok=True)

    filepath = os.path.join(base_path, f"config_{name}.json")
    with open(filepath, "w") as f:
        json.dump(config, f)


def read_data(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data