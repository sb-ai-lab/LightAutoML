from clearml import Task
from clearml import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import logging
import pandas as pd
import numpy as np
import os
from copy import copy
import shap
import matplotlib.pyplot as plt

from lightautoml.reader.base import PandasToPandasReader
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.automl.base import AutoML

from lightautoml.tasks import Task as LAMA_Task
import lightgbm as lgb
from lightautoml.addons.interpretation import SSWARM


logging.basicConfig(level=logging.INFO, filename="logs/py_log.log",filemode="w")

metric = {"multiclass":"crossentropy",
          "reg":"mse",
          "binary":"logloss"}

objective = {"multiclass":"multiclass",
          "reg":"regression",
          "binary":"binary"}
model_output = {"multiclass":"probability",
                "binary":"probability",
                "reg":"raw"}

BOOST_PARAMS = {
            "task": "train",
            "learning_rate": 0.05,
            "num_leaves": 128,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "max_depth": -1,
            "verbosity": -1,
            "reg_alpha": 1,
            "reg_lambda": 0.0,
            "min_split_gain": 0.0,
            "zero_as_missing": False,
            "num_threads": 4,
            "max_bin": 255,
            "min_data_in_bin": 3,
            "num_trees": 3000,
            "early_stopping_rounds": 100,
            "random_state": 42,
            }
    
def main(signature):
    dataset = Dataset.get(dataset_project="Datasets_with_metadata",
                          dataset_name=signature[0], dataset_tags=[signature[1]])
    dataset_local_path = dataset.get_local_copy()

    with open(os.path.join(dataset_local_path, "task_type.txt"), "r") as f:
        task_type = f.readline()
        
    if task_type == "multiclass": # multiclass not supported for now
        return
    
    train = pd.read_csv(os.path.join(dataset_local_path, "train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(dataset_local_path, "test.csv"), index_col=0)
    
    boost_params = copy(BOOST_PARAMS)
    target_col = train.columns[-1]
    if task_type != "reg":
        le = LabelEncoder()
        train[target_col] = le.fit_transform(train[target_col])
        test[target_col] = le.fit_transform(test[target_col])
        
    if task_type == "multiclass":
        boost_params.update({"num_classes":len(le.classes_)})
        
    boost_params.update({"objective": objective[task_type]})
    
    train = train.sample(min(25000, train.shape[0]), random_state=77).select_dtypes(include=[float, int])
    test = test.sample(min(10000, test.shape[0]), random_state=77).select_dtypes(include=[float, int])
    
    train.columns = ["Feature {}".format(i) for i in range(train.shape[1] - 1)] + [target_col]
    test.columns = ["Feature {}".format(i) for i in range(train.shape[1] - 1)] + [target_col]
    
    cml_task = Task.init(project_name="SHAP", task_name=signature[0] + "_plots", auto_connect_frameworks=False)
    logger = cml_task.get_logger()

            
    # if task_type != "reg":
    #     train, val = train_test_split(train, stratify=train[target_col], random_state=77)
        
    # else:
    train, val = train_test_split(train, random_state=77)
        
    task = LAMA_Task(task_type, metric=metric[task_type])
    
    # fit LightAutoML
    reader = PandasToPandasReader(task, random_state=77, advanced_roles=False)
    model1 = BoostLGBM(boost_params, freeze_defaults=True)
    pipeline_lvl1 = MLPipeline([model1])
    automl = AutoML(reader=reader, levels=[[pipeline_lvl1]], skip_conn=False)

    automl.fit_predict(train, valid_data=val, roles={'target': target_col}, verbose=4)
    
    X_train, X_val, X_test = train.drop(columns=target_col), val.drop(columns=target_col), test.drop(columns=target_col)
    y_train, y_val, y_test = train[target_col], val[target_col], test[target_col]
    
    # fit LGBM
    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(params=boost_params, train_set=d_train, valid_sets=[d_val])
    print(model_output[task_type])
    explainer_tree = shap.TreeExplainer(model, data=X_test.sample(500, random_state=77),
                                        feature_perturbation="interventional", model_output=model_output[task_type],
                                        check_additivity=False)
    shap_values_tree = explainer_tree.shap_values(X_test, check_additivity=False)
    if len(shap_values_tree.shape) == 3:
        shap_values_tree = shap_values_tree[-1]
        
    # map tree shap values
    shap_values_tree = pd.DataFrame(shap_values_tree, columns=X_test.columns)[automl.collect_used_feats()].to_numpy()

    # find seeded sswarm shapley values
    shap_seeds = np.empty((20, test.shape[0], len(automl.collect_used_feats())))
    for i in range(shap_seeds.shape[0]):
        explainer_sswarm = SSWARM(automl, random_state=i)
        shap_values_sswarm = explainer_sswarm.shap_values(test, T=500, verbose=False)
        if len(shap_values_sswarm.shape) == 3:
            shap_seeds[i] = shap_values_sswarm[-1]
        else:
            shap_seeds[i] = shap_values_sswarm
    
    shap_values_sswarm_avg_5 = np.mean(shap_seeds[:5], axis=0)
    shap_values_sswarm_avg_10 = np.mean(shap_seeds[:10], axis=0)
    shap_values_sswarm_avg_20 = np.mean(shap_seeds, axis=0)
    
    # Report errors
    logger.report_single_value("MSE default vs. one sswarm", mean_squared_error(shap_values_tree.flatten(), shap_seeds[0].flatten(), squared=False))
    logger.report_single_value("MSE default vs. sswarm avg 5", mean_squared_error(shap_values_tree.flatten(), shap_values_sswarm_avg_5.flatten(), squared=False))
    logger.report_single_value("MSE default vs. sswarm avg 10", mean_squared_error(shap_values_tree.flatten(), shap_values_sswarm_avg_10.flatten(), squared=False))
    logger.report_single_value("MSE default vs. sswarm avg 20", mean_squared_error(shap_values_tree.flatten(), shap_values_sswarm_avg_20.flatten(), squared=False))
    
    # Report summary plots
    plt_tree = shap.summary_plot(shap_values_tree, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="Tree", figure=plt_tree, series=" ", report_image=True, report_interactive=False)
    plt.show()
    
    plt_1 = shap.summary_plot(shap_seeds[0], X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 1", figure=plt_1, series=" ", report_image=True, report_interactive=False)
    plt.show()
    
    plt_5 = shap.summary_plot(shap_values_sswarm_avg_5, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 5", figure=plt_5, series=" ", report_image=True, report_interactive=False)
    plt.show()
    
    plt_10 = shap.summary_plot(shap_values_sswarm_avg_10, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 10", figure=plt_10, series=" ", report_image=True, report_interactive=False)
    plt.show()
    
    plt_20 = shap.summary_plot(shap_values_sswarm_avg_20, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 20", figure=plt_20, series=" ", report_image=True, report_interactive=False)
    plt.show()
    
    # report feature force plots
    num_rows = len(explainer_sswarm.used_feats)
    fig, ax = plt.subplots(num_rows, 5, figsize=(30, 60))
    ax[0][0].set_title('Tree')
    ax[0][1].set_title('SSWARM 1')
    ax[0][2].set_title('SSWARM 5')
    ax[0][3].set_title('SSWARM 10')
    ax[0][4].set_title('SSWARM 20')
    for i, feature in enumerate(explainer_sswarm.used_feats):
        shap.dependence_plot(feature, shap_values_tree, X_test[explainer_sswarm.used_feats], ax=ax[i][0], show=False, interaction_index=explainer_sswarm.used_feats[0])
        shap.dependence_plot(feature, shap_seeds[0], X_test[explainer_sswarm.used_feats], ax=ax[i][1], show=False, interaction_index=explainer_sswarm.used_feats[0])
        shap.dependence_plot(feature, shap_values_sswarm_avg_5, X_test[explainer_sswarm.used_feats], ax=ax[i][2], show=False, interaction_index=explainer_sswarm.used_feats[0])
        shap.dependence_plot(feature, shap_values_sswarm_avg_10, X_test[explainer_sswarm.used_feats], ax=ax[i][3], show=False, interaction_index=explainer_sswarm.used_feats[0])
        shap.dependence_plot(feature, shap_values_sswarm_avg_20, X_test[explainer_sswarm.used_feats], ax=ax[i][4], show=False, interaction_index=explainer_sswarm.used_feats[0])
    
    logger.report_matplotlib_figure(
        title="Force plots comparison", figure=fig, series=" ", report_image=True, report_interactive=False)
    logger.flush()
    cml_task.close()
if __name__ == "__main__":  
    dataset_signatures = set((i["name"], i["tags"][1 - i["tags"].index("kaggle")])
                    for i in Dataset.list_datasets(dataset_project="Datasets_with_metadata",
                                                    tags=["kaggle"]))
    for signature in dataset_signatures:
        logging.info("Started working on the dataset {0}".format(signature))
        main(signature)
        
