from clearml import Task
from clearml import Dataset
from typing import cast
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
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import time
from joblib import Parallel
from joblib import delayed

from lightautoml.reader.base import PandasToPandasReader
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.automl.base import AutoML
from lightautoml.reader.tabular_batch_generator import read_data, read_batch
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.dataset.utils import concatenate

from lightautoml.tasks import Task as LAMA_Task
import lightgbm as lgb
from lightautoml.addons.interpretation import SSWARM


logging.basicConfig(level=logging.INFO, filename="logs/py_log.log",filemode="w")

class SSWARM_iters(SSWARM):
    def shap_values(
        self, data, feature_names=None, T=500, n_jobs=1, verbose=False, val=None):
        """Computes shapley values consistent with the SHAP interface.

        Args:
            data: A matrix of samples on which to explain the model's output.
            feature_names: Feature names for automl prediction when data is not pd.DataFrame .
            T: Number of iterations.
            n_jobs: Number of parallel workers to execute automl.predict() .

        Returns:
            (# classes x # samples x # features) array of shapley values.
            (# samples x # features) in regression case.

        """
        self.num_obs = data.shape[0]
        if self.num_obs < 2:
            raise ValueError(
                """Too small number of observations.
                             Input data must contatin at least 2 observations."""
            )

        self.feature_names = feature_names
        if isinstance(data, pd.DataFrame) and feature_names is None:
            self.feature_names = data.columns.values

        self.data = np.array(data)
        self.feature_names = np.array(self.feature_names)

        # store prediction on all features v(N)
        v_N = self.v(self.data, n_jobs=1)
        self.expected_value = np.mean(v_N, axis=0)

        # initializing arrays for main variables
        # size is (num_obs x n x n x num_outputs)
        self.phi_plus = np.zeros((self.data.shape[0], self.n, self.n, self.n_outputs))
        self.phi_minus = np.zeros((self.data.shape[0], self.n, self.n, self.n_outputs))
        self.c_plus = np.zeros((self.data.shape[0], self.n, self.n, self.n_outputs))
        self.c_minus = np.zeros((self.data.shape[0], self.n, self.n, self.n_outputs))
        self.overall_mae = []
        self.step_mae = []

        if self.n > 3: # if n <= 3 -> we have already calculated all Shapley values exactly
            # initialization of \tilde{P}
            PMF = self.define_probability_distribution(self.n)
            if len(PMF) == 1:
                PMF = np.array([1])

            # First stage: collect updates
            # Updates are stored as a list of 3 sets
            self.updates = []

            # exact calculations of phi for coalitions of size 1, n-1, n
            self.exactCalculation()

            # warm up stages
            self.warmUp("plus")
            self.warmUp("minus")

            t = len(self.updates)

            # collect general updates
            for i in range(t, T):
                # draw the size of coalition from distribution P
                s_t = self.rng.choice(np.arange(2, self.n - 1), p=PMF)
                # draw the random coalition with size s_t
                # A_t = set(self.rng.choice([list(i) for i in combinations(self.N, s_t)]))
                A_t = self.draw_random_combination(self.N, s_t)
                # store combination
                self.updates.append([A_t, A_t, self.N.difference(A_t)])

            # Second stage: make updates

            # Num of the predictions made at once
            batch_size = 2_000_000
            if self.num_obs > batch_size:
                raise ValueError("Decrease the input number of observations")

            phi_prev = np.sum(self.phi_plus - self.phi_minus, axis=2) / self.n
            n_updates_per_round = batch_size // self.num_obs
            bar = tqdm(total=2 * T)
            cnt = 0
            for i in range(0, T, n_updates_per_round):
                pred_data = np.empty((n_updates_per_round * self.num_obs, self.data.shape[1]), dtype=np.object)

                # prepare the data
                iter_updates = self.updates[i : i + n_updates_per_round]
                for j, comb in enumerate(iter_updates):
                    A = comb[0]
                    A_plus = comb[1]
                    A_minus = comb[2]

                    temp = copy(self.data)
                    for col in self.N.difference(A):
                        # map column number from the used features space to the overall features space
                        mapped_col = np.where(np.array(self.feature_names) == self.used_feats[col])[0][0]
                        # shuffle mapped column
                        temp[:, mapped_col] = self.rng.permutation(temp[:, mapped_col])

                    pred_data[j * self.num_obs : (j + 1) * self.num_obs] = temp
                    bar.update(n=1)

                # make predictions
                v = self.v(pred_data, n_jobs=n_jobs)

                # make updates
                for j, comb in enumerate(iter_updates):
                    A = comb[0]
                    A_plus = comb[1]
                    A_minus = comb[2]

                    v_t = v[j * self.num_obs : (j + 1) * self.num_obs]
                    self.update(A, A_plus, A_minus, v_t)
                    
                    # compute mse since last update
                    phi_new = np.sum(self.phi_plus - self.phi_minus, axis=2) / self.n
                    mae_step = mean_absolute_error(phi_new[:, :, -1].flatten(),
                                                    phi_prev[:, :,-1].flatten())
                    self.step_mae.append(mae_step)
                    phi_prev = phi_new
                    
                    PHI = np.sum(phi_new, axis=1)
                    correction = ((v_N - PHI) / self.n).repeat(self.n, axis=0).reshape(len(phi_new), self.n, self.n_outputs)
                    phi_new = phi_new + correction - self.expected_value / self.n
                    phi_new = np.transpose(phi_new, axes=[2, 0, 1])
                    # if phi_new.shape[0] == 1:  # regression
                    #     phi_new = phi_new[0]
                        
                    mae_overall = mean_absolute_error(phi_new[-1].flatten(),
                                                    val.flatten())
                    
                    self.overall_mae.append(mae_overall)
                    

                    # if verbose:
                    #     print("MSE change since last update:", mse)
                    #     print("Overall MSE change", mse_overall)

                    # cnt += 1
                    # if cnt > 6:
                    #     break
                    bar.update(n=1)

            bar.close()

        phi = np.sum(self.phi_plus - self.phi_minus, axis=2) / self.n
        PHI = np.sum(phi, axis=1)

        # normalize phi to sum to the predicted outcome
        # and substract expected value to be consistent with SHAP python library
        correction = ((v_N - PHI) / self.n).repeat(self.n, axis=0).reshape(len(phi), self.n, self.n_outputs)
        phi = phi + correction - self.expected_value / self.n

        phi = np.transpose(phi, axes=[2, 0, 1])
        if phi.shape[0] == 1:  # regression
            phi = phi[0]
            self.expected_value = self.expected_value[0]
        return phi
    
class AutoMLQuick(AutoML):
    cpu_limit = 1
    def predict_default(
        self,
        data,
        features_names = None,
        return_all_predictions = None,
    ):
        """Predict with automl on new dataset.

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
                if cannot be inferred from `train_data`.
            return_all_predictions: if True,
                returns all model predictions from last level

        Returns:
            Dataset with predictions.

        """
        dataset = self.reader.read(data, features_names=features_names, add_array_attrs=False)

        for n, level in enumerate(self.levels, 1):
            # check if last level

            level_predictions = []
            for _n, ml_pipe in enumerate(level):
                level_predictions.append(ml_pipe.predict(dataset))

            if n != len(self.levels):

                level_predictions = concatenate(level_predictions)

                if self.skip_conn:

                    try:
                        # convert to initital dataset type
                        level_predictions = dataset.from_dataset(level_predictions)
                    except TypeError:
                        raise TypeError(
                            "Can not convert prediction dataset type to input features. Set skip_conn=False"
                        )
                    dataset = concatenate([level_predictions, dataset])
                else:
                    dataset = level_predictions
            else:
                if (return_all_predictions is None and self.return_all_predictions) or return_all_predictions:
                    return concatenate(level_predictions)
                return self.blender.predict(level_predictions)
    def predict(
        self,
        data,
        features_names=None,
        batch_size=None,
        n_jobs = 1,
        return_all_predictions = None,
    ):
        """Get dataset with predictions.

        Almost same as :meth:`lightautoml.automl.base.AutoML.predict`
        on new dataset, with additional features.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`. For example,
              ``{'data': X...}``. In this case roles are optional,
              but `train_features` and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Parallel inference - you can pass ``n_jobs`` to speedup
        prediction (requires more RAM).
        Batch_inference - you can pass ``batch_size``
        to decrease RAM usage (may be longer).

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
                if cannot be inferred from `train_data`.
            batch_size: Batch size or ``None``.
            n_jobs: Number of jobs.
            return_all_predictions: if True,
                returns all model predictions from last level

        Returns:
            Dataset with predictions.

        """
        #read_csv_params = self._get_read_csv_params()
        read_csv_params = None
        if batch_size is None and n_jobs == 1:
            data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)
            pred = self.predict_default(data, features_names, return_all_predictions)
            return cast(NumpyDataset, pred)

        self.cpu_limit = n_jobs

        data_generator = read_batch(
            data,
            features_names,
            n_jobs=n_jobs,
            batch_size=batch_size,
            read_csv_params=read_csv_params,
        )

        if n_jobs == 1:
            res = [self.predict(df, features_names, return_all_predictions) for df in data_generator]
        else:
            # TODO: Check here for pre_dispatch param
            with Parallel(n_jobs, pre_dispatch=len(data_generator) + 1) as p:
                res = p(delayed(self.predict)(df, features_names, return_all_predictions) for df in data_generator)

        res = NumpyDataset(
            np.concatenate([x.data for x in res], axis=0),
            features=res[0].features,
            roles=res[0].roles,
        )

        return res

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
    
def main(name_version):
    dataset = Dataset.get(dataset_project="Datasets_with_metadata",
                          dataset_name=name_version[0], dataset_version=name_version[1])
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
    
    cml_task = Task.init(project_name="SHAP", task_name=name_version[0] + "_" + name_version[1], auto_connect_frameworks=False)
    logger = cml_task.get_logger()
    
    # Report dataset stats
    logger.report_single_value("num feats", test.shape[1] - 1)
    logger.report_single_value("num test obs", test.shape[0])

            
    # if task_type != "reg":
    #     train, val = train_test_split(train, stratify=train[target_col], random_state=77)
        
    # else:
    train, val = train_test_split(train, random_state=77)
        
    task = LAMA_Task(task_type, metric=metric[task_type])
    
    # fit LightAutoML
    reader = PandasToPandasReader(task, random_state=77, advanced_roles=False)
    model1 = BoostLGBM(boost_params, freeze_defaults=True)
    pipeline_lvl1 = MLPipeline([model1])
    automl = AutoMLQuick(reader=reader, levels=[[pipeline_lvl1]], skip_conn=False)

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

    explainer_sswarm = SSWARM_iters(automl, random_state=77)
    shap_values_sswarm = explainer_sswarm.shap_values(test, T=2000, verbose=False, val=shap_values_tree, n_jobs=5)
    
    fig1, ax = plt.subplots(1, 1)
    ax.plot(explainer_sswarm.overall_mae)
    logger.report_matplotlib_figure(
        title="sswarm vs. shap", figure=fig1, series=" ", report_image=True, report_interactive=False)
    
    fig2, ax = plt.subplots(1, 1)
    ax.plot(explainer_sswarm.step_mae)
    logger.report_matplotlib_figure(
        title="step change", figure=fig2, series=" ", report_image=True, report_interactive=False)
    
    
    
    # find seeded sswarm shapley values
    if test.shape[1] - 1 > 20:
        num_iters = 1000
    arr_of_time =[]
    shap_seeds = np.empty((20, test.shape[0], len(automl.collect_used_feats())))
    start = time.time()
    for i in range(shap_seeds.shape[0]):
        explainer_sswarm = SSWARM(automl, random_state=i)
        shap_values_sswarm = explainer_sswarm.shap_values(test, T=1000, verbose=False, n_jobs=5)
        if len(shap_values_sswarm.shape) == 3:
            shap_seeds[i] = shap_values_sswarm[-1]
        else:
            shap_seeds[i] = shap_values_sswarm
        if i in [0, 4, 9, 19]:
            checkpoint = time.time()
            arr_of_time.append(checkpoint - start)
    
    shap_values_sswarm_avg_5 = np.mean(shap_seeds[:5], axis=0)
    shap_values_sswarm_avg_10 = np.mean(shap_seeds[:10], axis=0)
    shap_values_sswarm_avg_20 = np.mean(shap_seeds, axis=0)
    
    # Report errors
    logger.report_single_value("MAE default vs. one sswarm", mean_squared_error(shap_values_tree.flatten(), shap_seeds[0].flatten(), squared=False))
    logger.report_single_value("MAE default vs. sswarm avg 5", mean_squared_error(shap_values_tree.flatten(), shap_values_sswarm_avg_5.flatten(), squared=False))
    logger.report_single_value("MAE default vs. sswarm avg 10", mean_squared_error(shap_values_tree.flatten(), shap_values_sswarm_avg_10.flatten(), squared=False))
    logger.report_single_value("MAE default vs. sswarm avg 20", mean_squared_error(shap_values_tree.flatten(), shap_values_sswarm_avg_20.flatten(), squared=False))
    
    # Report time
    logger.report_single_value("sswarm 1", arr_of_time[0])
    logger.report_single_value("sswarm 5", arr_of_time[1])
    logger.report_single_value("sswarm 10", arr_of_time[2])
    logger.report_single_value("sswarm 20", arr_of_time[3])
    
    # Report summary plots
    shap.summary_plot(shap_values_tree, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="Tree", figure=plt, series=" ", report_image=True, report_interactive=False)
    plt.show(); plt.clf()
    
    shap.summary_plot(shap_seeds[0], X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 1", figure=plt, series=" ", report_image=True, report_interactive=False)
    plt.show(); plt.clf()
    
    shap.summary_plot(shap_values_sswarm_avg_5, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 5", figure=plt, series=" ", report_image=True, report_interactive=False)
    plt.show(); plt.clf()
    
    shap.summary_plot(shap_values_sswarm_avg_10, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 10", figure=plt, series=" ", report_image=True, report_interactive=False)
    plt.show(); plt.clf()
    
    shap.summary_plot(shap_values_sswarm_avg_20, X_test[explainer_sswarm.used_feats], show=False, plot_type="dot", color_bar=False)
    logger.report_matplotlib_figure(
        title="SSWARM 20", figure=plt, series=" ", report_image=True, report_interactive=False)
    plt.show(); plt.clf()
    
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
    
    dataset_name_version = [[i["name"], i["version"]]
                        for signature in dataset_signatures
                        for i in Dataset.list_datasets(dataset_project="Datasets_with_metadata",
                                                       partial_name=signature[0], tags=[signature[1]])[:2]]
    for name_version in dataset_name_version:
        logging.info("Started working on the dataset {0}".format(name_version))
        main(name_version)
        
