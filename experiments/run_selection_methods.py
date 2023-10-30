import signal
import numpy as np
import argparse
import os
import clearml
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from utils import Timer
from utils import install_lightautoml

install_lightautoml()

def metric(y_true, y_pred, task, reader):
    if task == "reg":
        return mean_absolute_error(y_true, y_pred)
    if task == "multiclass":
        mapping = reader.class_mapping
        if mapping is not None:
            oredered_labels = list({key: val for (key, val) in sorted(mapping.items())}.values())
            y_pred = y_pred[:, oredered_labels]
        return roc_auc_score(y_true.values, y_pred, average="macro", multi_class="ovo")
    if task == "binary":
        mapping = reader.class_mapping
        y_inv = np.ones(y_pred.shape) - y_pred
        y_pred = np.append(y_inv, y_pred, axis=1)
        if mapping is not None:
            oredered_labels = list({key: val for (key, val) in sorted(mapping.items())}.values())
            y_pred = y_pred[:, oredered_labels]
        return roc_auc_score(y_true, y_pred[:, 1])
    

class TimeoutException(BaseException): pass

def signal_handler(signum, frame):
    raise TimeoutException("Timed out!")

signal.signal(signal.SIGALRM, signal_handler)

config_path = {}
config_path["NoSelection"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_no_selection.yml"
config_path["CutoffGain"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_mode_1.yml"
config_path["IterativeForward"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_mode_2_forward.yml"
config_path["IterativeBackward"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_mode_2_backward.yml"
config_path["IterativeForwardCrossVal"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_mode_2_forward_crossval.yml"
config_path["IterativeBackwardCrossVal"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_mode_2_backward_crossval.yml"
config_path["relief"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_relief.yml"
# config_path["SURF"] = "/home/pbelonovskiy/LAMA_selection/presets/tabular_config_SURF.yml"
config_path["MultiSURF"]= "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_MultiSURF.yml"
config_path["fcbf"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_FCBF.yml"
config_path["MIR"] = "../lightautoml/automl/presets/tabular_configs/presets/tabular_config_MIR.yml"

RANDOM_SEED = 777    


# TARGET_NAME = "class"


def main(dataset_name: str, dataset_version: str,
         dataset_project: str, cpu_limit: int,
         memory_limit: int):  # noqa D103
    cml_task = clearml.Task.get_task(clearml.config.get_remote_task_id())
    logger = cml_task.get_logger()

    dataset = clearml.Dataset.get(dataset_id=None, dataset_project=dataset_project,
                                  dataset_name=dataset_name, dataset_version=dataset_version)
    dataset_local_path = dataset.get_local_copy()

    with open(os.path.join(dataset_local_path, "task_type.txt"), "r") as f:
        task_type = f.readline()
    train = pd.read_csv(os.path.join(dataset_local_path, "train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(dataset_local_path, "test.csv"), index_col=0)
    TARGET_NAME = train.columns[-1]

    for key in config_path:
        if task_type == "reg" and key == "fcbf":
            continue

        if task_type in ["binary", "multiclass"]  and key == "MIR":
            continue   

        time_limit = 30
        
        np.random.seed(RANDOM_SEED)
        automl = TabularAutoML(
            task=Task(task_type),
            cpu_limit=cpu_limit,
            config_path=config_path[key],
            timeout=time_limit,
            memory_limit=memory_limit,
            reader_params = {'random_state': RANDOM_SEED})
        
        signal.alarm(time_limit+100)
        cml_task.connect(automl)

        with Timer() as timer_training:
            oof_predictions = automl.fit_predict(train, roles={"target": TARGET_NAME}, verbose=0)

        with Timer() as timer_predict:
            test_predictions = automl.predict(test)
        signal.alarm(0)

        weight_of_LR = None
        num_used_feats = train.shape[1] - 1
        result_oof = metric(train[TARGET_NAME], oof_predictions.data, task_type, automl.reader)
        result_ho = metric(test[TARGET_NAME], test_predictions.data, task_type, automl.reader)
        selector_idc = len(automl.levels[0]) - 1
        num_used_feats = len(automl.levels[0][selector_idc].pre_selection.selected_features)
        if 'Lvl_0_Pipe_0_Mod_0_LinearL2' in automl.collect_model_stats():
            weight_of_LR = automl.blender.wts[0]
        output_table = pd.DataFrame([[dataset_name, dataset_version, task_type, key, result_oof,
                                      result_ho, timer_training, train.shape[0], train.shape[1]-1,
                                      num_used_feats, weight_of_LR]],
                                      columns=["dataset_name", "dataset_version",
                                               "task_type", "selection_method",
                                               "result_oof", "result_ho", "training_time",
                                               "num_train_obs", "num_test_obs","num_used_feats",
                                               "weight_of_LR"])
        
        logger.report_table(title="Results", series='pandas DataFrame', table_plot=output_table)
        logger.flush()
        print("Working on the selection type {} completed".format(key))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="dataset name or id", default="sampled_app_train")
    parser.add_argument("--dataset_project", type=str, help="dataset project", default="Datasets_with_metadata")
    parser.add_argument("--dataset_version", type=str, help="dataset version", default=None)
    parser.add_argument("--cpu_limit", type=int, help="", default=8)
    parser.add_argument("--memory_limit", type=int, help="", default=16)
    args = parser.parse_args()

    main(dataset_name=args.dataset, dataset_project=args.dataset_project,
         dataset_version=args.dataset_version, cpu_limit=args.cpu_limit,
         memory_limit=args.memory_limit)
