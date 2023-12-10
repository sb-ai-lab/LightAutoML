from clearml import Task
from clearml import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import logging
import pandas as pd
import numpy as np
import os
from copy import copy
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        batch_size = 500_000
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
    
    if len(train.columns) <= 4:
        return
    
    cml_task = Task.init(project_name="SHAP", task_name=signature[0] + "_iters", auto_connect_frameworks=False)
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

    explainer_sswarm = SSWARM_iters(automl, random_state=77)
    shap_values_sswarm = explainer_sswarm.shap_values(test, T=2000, verbose=False, val=shap_values_tree)
    
    fig1, ax = plt.subplots(1, 1)
    ax.plot(explainer_sswarm.overall_mae)
    logger.report_matplotlib_figure(
        title="sswarm vs. shap", figure=fig1, series=" ", report_image=True, report_interactive=False)
    
    fig2, ax = plt.subplots(1, 1)
    ax.plot(explainer_sswarm.step_mae)
    logger.report_matplotlib_figure(
        title="step change", figure=fig2, series=" ", report_image=True, report_interactive=False)
    
    logger.flush()
    cml_task.close()
if __name__ == "__main__":  
    dataset_signatures = set((i["name"], i["tags"][1 - i["tags"].index("kaggle")])
                    for i in Dataset.list_datasets(dataset_project="Datasets_with_metadata",
                                                    tags=["kaggle"]))
    for signature in dataset_signatures:
        logging.info("Started working on the dataset {0}".format(signature))
        main(signature)
        
