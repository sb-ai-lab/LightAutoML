"""Tabular presets (GPU version)."""

import logging
import os
from copy import copy
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

import cupy as cp
import torch
from pandas import DataFrame

from lightautoml.automl.gpu.blend_gpu import MeanBlenderGPU
from lightautoml.automl.gpu.blend_gpu import WeightedBlenderGPU
from lightautoml.automl.presets.base import upd_params
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.ml_algo.gpu.boost_cb_gpu import BoostCBGPU
from lightautoml.ml_algo.gpu.boost_pb_gpu import BoostPB
from lightautoml.ml_algo.gpu.boost_xgb_gpu import BoostXGB
from lightautoml.ml_algo.gpu.boost_xgb_gpu import BoostXGB_dask
from lightautoml.ml_algo.gpu.linear_gpu import LinearLBFGSGPU
from lightautoml.ml_algo.tuning.gpu.optuna_gpu import GpuQueue
from lightautoml.ml_algo.tuning.gpu.optuna_gpu import OptunaTunerGPU
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.gpu.lgb_pipeline_gpu import LGBAdvancedPipelineGPU
from lightautoml.pipelines.features.gpu.lgb_pipeline_gpu import LGBSimpleFeaturesGPU
from lightautoml.pipelines.features.gpu.linear_pipeline_gpu import LinearFeaturesGPU
from lightautoml.pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline
from lightautoml.pipelines.selection.base import ComposedSelector
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import ModelBasedImportanceEstimator
from lightautoml.pipelines.selection.permutation_importance_based import NpIterativeFeatureSelector
from lightautoml.pipelines.selection.permutation_importance_based import NpPermutationImportanceEstimator
from lightautoml.reader.gpu.cudf_reader import CudfReader
from lightautoml.reader.gpu.daskcudf_reader import DaskCudfReader
from lightautoml.reader.tabular_batch_generator import ReadableToDf
from lightautoml.reader.tabular_batch_generator import read_batch
from lightautoml.reader.tabular_batch_generator import read_data
from lightautoml.tasks import Task

GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]

_base_dir = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


class TabularAutoMLGPU(TabularAutoML):
    """
    GPU version of TabularAutoML.

    No lightgbm support. Algos that run on GPU: xgboost, torch based algorithms, pyboost, catboost.
    """

    _default_config_path = "gpu/tabular_gpu_config.yml"

    # set initial runtime rate guess for first level models
    _time_scores = {
        "xgb": 2,
        "xgb_tuned": 6,
        "linear_l2": 0.7,
        "cb": 1,
        "cb_tuned": 3,
        "pb": 1,
        "pb_tuned": 3,
    }

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = "all",
        timing_params: Optional[dict] = None,
        config_path: Optional[str] = None,
        general_params: Optional[dict] = None,
        reader_params: Optional[dict] = None,
        read_csv_params: Optional[dict] = None,
        nested_cv_params: Optional[dict] = None,
        tuning_params: Optional[dict] = None,
        selection_params: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        cb_params: Optional[dict] = None,
        pb_params: Optional[dict] = None,
        linear_l2_params: Optional[dict] = None,
        gbm_pipeline_params: Optional[dict] = None,
        linear_pipeline_params: Optional[dict] = None,
        client=None,
    ):
        super(TabularAutoMLGPU.__bases__[0], self).__init__(
            task, timeout, memory_limit, cpu_limit, gpu_ids, timing_params, config_path
        )

        # upd manual params
        for name, param in zip(
            [
                "general_params",
                "reader_params",
                "read_csv_params",
                "nested_cv_params",
                "tuning_params",
                "selection_params",
                "xgb_params",
                "cb_params",
                "pb_params",
                "linear_l2_params",
                "gbm_pipeline_params",
                "linear_pipeline_params",
            ],
            [
                general_params,
                reader_params,
                read_csv_params,
                nested_cv_params,
                tuning_params,
                selection_params,
                xgb_params,
                cb_params,
                pb_params,
                linear_l2_params,
                gbm_pipeline_params,
                linear_pipeline_params,
            ],
        ):
            if param is None:
                param = {}
            self.__dict__[name] = upd_params(self.__dict__[name], param)
        self.client = client

    def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False):

        if torch.cuda.device_count() == 1:
            self.general_params["parallel_folds"] = False
            self.xgb_params["parallel_folds"] = False
            self.cb_params["parallel_folds"] = False
            self.pb_params["parallel_folds"] = False
            self.linear_l2_params["parallel_folds"] = False

        else:
            try:
                val = self.general_params["parallel_folds"]
            except KeyError:
                val = False

            try:
                res = self.xgb_params["parallel_folds"]
            except KeyError:
                self.xgb_params["parallel_folds"] = val
            try:
                res = self.cb_params["parallel_folds"]
            except KeyError:
                self.cb_params["parallel_folds"] = val
            try:
                res = self.pb_params["parallel_folds"]
            except KeyError:
                self.pb_params["parallel_folds"] = val

            res = self.linear_l2_params.get("parallel_folds")
            if res is None:
                self.linear_l2_params["parallel_folds"] = val
            else:
                self.linear_l2_params["parallel_folds"] = res

        length = train_data.shape[0]

        # infer optuna tuning iteration based on dataframe len
        if self.tuning_params["max_tuning_iter"] == "auto":
            if length < 10000:
                self.tuning_params["max_tuning_iter"] = 100
            elif length < 30000:
                self.tuning_params["max_tuning_iter"] = 50
            elif length < 100000:
                self.tuning_params["max_tuning_iter"] = 10
            else:
                self.tuning_params["max_tuning_iter"] = 5

        if self.general_params["use_algos"] == "auto":
            self.general_params["use_algos"] = [["xgb", "xgb_tuned", "linear_l2", "cb", "cb_tuned"]]
            if self.task.name == "multiclass" and multilevel_avail:
                self.general_params["use_algos"].append(["linear_l2", "cb"])
            if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
                self.general_params["use_algos"] = [["xgb", "xgb_tuned", "linear_l2", "pb", "pb_tuned"]]

        if not self.general_params["nested_cv"]:
            self.nested_cv_params["cv"] = 1

        # check gpu to use catboost
        if self.task.device == "gpu" or self.cb_params["parallel_folds"]:
            gpu_cnt = 1
        else:
            gpu_cnt = torch.cuda.device_count()

        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                cur_gpu_ids = ",".join(list(map(str, range(gpu_cnt))))
            else:
                cur_gpu_ids = copy(gpu_ids)
            self.cb_params["default_params"]["task_type"] = "GPU"
            self.cb_params["default_params"]["devices"] = cur_gpu_ids.replace(",", ":")
            self.cb_params["gpu_ids"] = cur_gpu_ids.split(",")

        if self.task.device == "gpu" or self.xgb_params["parallel_folds"]:
            gpu_cnt = 1
        else:
            gpu_cnt = torch.cuda.device_count()

        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                cur_gpu_ids = ",".join(list(map(str, range(gpu_cnt))))
            else:
                cur_gpu_ids = copy(gpu_ids)

            self.xgb_params["gpu_ids"] = cur_gpu_ids.split(",")

        if self.task.device == "gpu" or self.linear_l2_params["parallel_folds"]:
            gpu_cnt = 1
        else:
            gpu_cnt = torch.cuda.device_count()

        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                cur_gpu_ids = ",".join(list(map(str, range(gpu_cnt))))
            else:
                cur_gpu_ids = copy(gpu_ids)

            self.linear_l2_params["gpu_ids"] = cur_gpu_ids.split(",")

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        self.cb_params["default_params"]["thread_count"] = min(
            self.cb_params["default_params"]["thread_count"], cpu_cnt
        )
        self.xgb_params["default_params"]["nthread"] = min(
            self.xgb_params["default_params"]["nthread"], cpu_cnt
        )
        self.reader_params["n_jobs"] = min(self.reader_params["n_jobs"], cpu_cnt)
        if not self.linear_l2_params["parallel_folds"] and self.task.device == "mgpu":
            self.reader_params["output"] = "mgpu"

    def get_time_score(self, n_level: int, model_type: str, nested: Optional[bool] = None):

        if nested is None:
            nested = self.general_params["nested_cv"]

        score = self._time_scores[model_type]

        if self.task.device == "gpu":
            num_gpus = 1
        else:
            num_gpus = torch.cuda.device_count()

        if model_type in ["cb", "cb_tuned"]:
            if self.cb_params["parallel_folds"]:
                score /= num_gpus
        if model_type in ["xgb", "xgb_tuned"]:
            if self.xgb_params["parallel_folds"]:
                score /= num_gpus
        if model_type in ["pb", "pb_tuned"]:
            if self.pb_params["parallel_folds"]:
                score /= num_gpus
        if model_type == "linear_l2":
            if self.linear_l2_params["parallel_folds"]:
                score /= num_gpus

        mult = 1
        if nested:
            if self.nested_cv_params["n_folds"] is not None:
                mult = self.nested_cv_params["n_folds"]
            else:
                mult = self.nested_cv_params["cv"]

        if n_level > 1:
            mult *= 0.8 if self.general_params["skip_conn"] else 0.1

        score = score * mult

        # lower score for catboost on gpu
        if (
            model_type in ["cb", "cb_tuned"]
            and self.cb_params["default_params"]["task_type"] == "GPU"
        ):
            score *= 0.5
        return score

    def get_selector(self, n_level: Optional[int] = 1) -> SelectionPipeline:
        selection_params = self.selection_params

        mode = selection_params["mode"]

        # create pre selection based on mode
        pre_selector = None
        if mode > 0:
            # if we need selector - define model
            selection_feats = LGBSimpleFeaturesGPU()

            # IF PYBOOST IS FASTER MAYBE SINGLE GPU SHOULD BE PB
            if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
                # timer will be useful to estimate time for next gbm runs
                time_score = self.get_time_score(n_level, "pb", False)
                sel_timer_0 = self.timer.get_task_timer("pb", time_score)
            else:
                time_score = self.get_time_score(n_level, "cb", False)
                sel_timer_0 = self.timer.get_task_timer("cb", time_score)
                selection_gbm = BoostCBGPU(timer=sel_timer_0, **self.cb_params)
            selection_gbm.set_prefix("Selector")

            if selection_params["importance_type"] == "permutation":
                importance = NpPermutationImportanceEstimator()
            else:
                importance = ModelBasedImportanceEstimator()

            pre_selector = ImportanceCutoffSelector(
                selection_feats,
                selection_gbm,
                importance,
                cutoff=selection_params["cutoff"],
                fit_on_holdout=selection_params["fit_on_holdout"],
            )
            if mode == 2:
                selection_feats = LGBSimpleFeaturesGPU()
                if (self.task.name == "multi:reg") or (self.task.name == "multilabel"):
                    time_score = self.get_time_score(n_level, "xgb", False)
                    sel_timer_1 = self.timer.get_task_timer("xgb", time_score)
                    if self.task.device == "gpu":
                        selection_gbm = BoostXGB(timer=sel_timer_1, **self.xgb_params)
                    elif self.task.device == "mgpu":
                        selection_gbm = BoostXGB_dask(
                            client=self.client, timer=sel_timer_1, **self.xgb_params
                        )
                else:
                    time_score = self.get_time_score(n_level, "cb", False)
                    sel_timer_1 = self.timer.get_task_timer("cb", time_score)
                    selection_gbm = BoostCBGPU(timer=sel_timer_1, **self.cb_params)
                selection_gbm.set_prefix("Selector")

                importance = NpPermutationImportanceEstimator()

                extra_selector = NpIterativeFeatureSelector(
                    selection_feats,
                    selection_gbm,
                    importance,
                    feature_group_size=selection_params["feature_group_size"],
                    max_features_cnt_in_result=selection_params[
                        "max_features_cnt_in_result"
                    ],
                )

                pre_selector = ComposedSelector([pre_selector, extra_selector])

        return pre_selector

    def get_linear(
        self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None
    ) -> NestedTabularMLPipeline:

        # linear model with l2
        time_score = self.get_time_score(n_level, "linear_l2")
        linear_l2_timer = self.timer.get_task_timer("reg_l2", time_score)
        linear_l2_model = LinearLBFGSGPU(timer=linear_l2_timer, **self.linear_l2_params)
        linear_l2_feats = LinearFeaturesGPU(
            output_categories=True, **self.linear_pipeline_params
        )

        linear_l2_pipe = NestedTabularMLPipeline(
            [linear_l2_model],
            force_calc=True,
            pre_selection=pre_selector,
            features_pipeline=linear_l2_feats,
            **self.nested_cv_params
        )
        return linear_l2_pipe

    def get_gbms(
        self,
        keys: Sequence[str],
        n_level: int = 1,
        pre_selector: Optional[SelectionPipeline] = None,
    ) -> NestedTabularMLPipeline:

        gbm_feats = LGBAdvancedPipelineGPU(**self.gbm_pipeline_params)

        ml_algos = []
        force_calc = []
        for key, force in zip(keys, [True, False, False, False]):
            tuned = "_tuned" in key
            algo_key = key.split("_")[0]
            time_score = self.get_time_score(n_level, key)
            gbm_timer = self.timer.get_task_timer(algo_key, time_score)
            if algo_key == "cb":
                gbm_model = BoostCBGPU(timer=gbm_timer, **self.cb_params)
            elif algo_key == "xgb":
                if self.task.device == "mgpu" and not self.xgb_params["parallel_folds"]:
                    gbm_model = BoostXGB_dask(
                        client=self.client, timer=gbm_timer, **self.xgb_params
                    )
                else:
                    gbm_model = BoostXGB(timer=gbm_timer, **self.xgb_params)
            elif algo_key == "pb":
                gbm_model = BoostPB(timer=gbm_timer, **self.pb_params)
            else:
                raise ValueError("Wrong algo key")

            if tuned:
                gbm_model.set_prefix("Tuned")
                if algo_key == "cb":
                    folds = self.cb_params["parallel_folds"]
                    if self.task.device == "gpu":
                        gpu_cnt = 1
                    else:
                        gpu_cnt = torch.cuda.device_count()
                elif algo_key == "xgb":
                    folds = self.xgb_params["parallel_folds"]
                    if self.task.device == "gpu":
                        gpu_cnt = 1
                    else:
                        gpu_cnt = torch.cuda.device_count()
                elif algo_key == "pb":
                    folds = self.pb_params["parallel_folds"]
                    if self.task.device == "gpu":
                        gpu_cnt = 1
                    else:
                        gpu_cnt = torch.cuda.device_count()
                else:
                    raise ValueError("Wrong algo key")

                if folds:

                    gbm_tuner = OptunaTunerGPU(
                        ngpus=gpu_cnt,
                        gpu_queue=GpuQueue(ngpus=gpu_cnt),
                        n_trials=self.tuning_params["max_tuning_iter"],
                        timeout=self.tuning_params["max_tuning_time"],
                        fit_on_holdout=self.tuning_params["fit_on_holdout"],
                    )
                else:
                    gbm_tuner = OptunaTuner(
                        n_trials=self.tuning_params["max_tuning_iter"],
                        timeout=self.tuning_params["max_tuning_time"],
                        fit_on_holdout=self.tuning_params["fit_on_holdout"],
                    )

                gbm_model = (gbm_model, gbm_tuner)
            ml_algos.append(gbm_model)
            force_calc.append(force)

        gbm_pipe = NestedTabularMLPipeline(
            ml_algos,
            force_calc,
            pre_selection=pre_selector,
            features_pipeline=gbm_feats,
            **self.nested_cv_params
        )

        return gbm_pipe

    def create_automl(self, **fit_args):
        """Create basic automl instance (GPU version).

        Args:
            **fit_args: Contain all information needed for creating automl.

        """
        train_data = fit_args["train_data"]
        multilevel_avail = fit_args["valid_data"] is None and fit_args["cv_iter"] is None

        self.infer_auto_params(train_data, multilevel_avail)

        num_data = train_data.shape[0] * train_data.shape[1]

        if num_data < 1e8 or self.task.device == "gpu":
            reader = CudfReader(task=self.task, **self.reader_params)
        else:
            if self.task.device != "cpu":
                reader = DaskCudfReader(task=self.task, **self.reader_params)
            else:
                raise ValueError("Device must be either gpu or mgpu to run on GPUs")

        pre_selector = self.get_selector()

        levels = []

        for n, names in enumerate(self.general_params["use_algos"]):
            lvl = []
            # regs
            if "linear_l2" in names:
                selector = None
                if "linear_l2" in self.selection_params["select_algos"] and (
                    self.general_params["skip_conn"] or n == 0
                ):
                    selector = pre_selector
                lvl.append(self.get_linear(n + 1, selector))

            gbm_models = [
                x
                for x in ["cb", "cb_tuned", "xgb", "xgb_tuned", "pb", "pb_tuned"]
                if x in names and x.split("_")[0] in self.task.losses
            ]

            if len(gbm_models) > 0:
                selector = None
                if "gbm" in self.selection_params["select_algos"] and (
                    self.general_params["skip_conn"] or n == 0
                ):
                    selector = pre_selector
                lvl.append(self.get_gbms(gbm_models, n + 1, selector))

            levels.append(lvl)

        # blend everything
        blender = WeightedBlenderGPU(
            max_nonzero_coef=self.general_params["weighted_blender_max_nonzero_coef"]
        )

        # initialize
        self._initialize(
            reader,
            levels,
            skip_conn=self.general_params["skip_conn"],
            blender=blender,
            return_all_predictions=self.general_params["return_all_predictions"],
            timer=self.timer,
        )

    def fit_predict(
        self,
        train_data: ReadableToDf,
        roles: Optional[dict] = None,
        train_features: Optional[Sequence[str]] = None,
        cv_iter: Optional[Iterable] = None,
        valid_data: Optional[ReadableToDf] = None,
        valid_features: Optional[Sequence[str]] = None,
        log_file: str = None,
        verbose: int = 0,
        return_numpy: bool = True,
    ) -> Union[GpuDataset, NumpyDataset]:
        """Fit and get prediction on validation dataset (GPU version).

        Additionally supports:
            - :class:`~cupy.ndarray`.
            - :class:`cudf.DataFrame`.
            - :class:`daskcudf.DataFrame`.
        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Optional features names, if can't
              be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example,
              :class:`~lightautoml.validation.gpu.gpu_iterators.HoldoutIteratorGPU`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features
              if cannot be inferred from `valid_data`.
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;
            log_file: Filename for writing logging messages. If log_file is specified,
            the messages will be saved in a the file. If the file exists, it will be overwritten.
            return_numpy: bool to return output in CPU or GPU
        Returns:
            Dataset with predictions on GPU or CPU. Call ``.data`` to get predictions array.
        """
        self.set_logfile(log_file)

        if roles is None:
            roles = {}
        read_csv_params = self._get_read_csv_params()
        train, upd_roles = read_data(
            train_data, train_features, self.cpu_limit, read_csv_params
        )
        if upd_roles:
            roles = {**roles, **upd_roles}
        if valid_data is not None:
            data, _ = read_data(
                valid_data, valid_features, self.cpu_limit, self.read_csv_params
            )

        oof_pred = super(TabularAutoMLGPU.__bases__[0], self).fit_predict(
            train, roles=roles, cv_iter=cv_iter, valid_data=valid_data, verbose=verbose
        )

        if return_numpy:
            return oof_pred.to_numpy()
        else:
            return oof_pred

    def predict(
        self,
        data: ReadableToDf,
        features_names: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        n_jobs: Optional[int] = 1,
        return_all_predictions: Optional[bool] = None,
        return_numpy: bool = True,
    ) -> Union[GpuDataset, NumpyDataset]:
        """Get dataset with predictions (GPU version).

        Additionally supports:
            - :class:`~cupy.ndarray`.
            - :class:`cudf.DataFrame`.
            - :class:`daskcudf.DataFrame`.

        Parallel inference is not available (n_jobs will be set to 1).

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
              if cannot be inferred from `train_data`.
            batch_size: Batch size or ``None``.
            n_jobs: Number of jobs (available only for consistency).
            return_all_predictions: if True,
              returns all model predictions from last level
            return_numpy: bool to return output in CPU or GPU
        Returns:
            Dataset with predictions on GPU or CPU. Call ``.data`` to get predictions array.
        """

        if n_jobs != 1:
            n_jobs = 1

        read_csv_params = self._get_read_csv_params()

        if batch_size is None:
            data, _ = read_data(data, features_names, self.cpu_limit, read_csv_params)

            pred = super(TabularAutoMLGPU.__bases__[0], self).predict(
                data, features_names, return_all_predictions
            )
            if return_numpy:
                return pred.to_numpy()
            else:
                return pred

        data_generator = read_batch(
            data,
            features_names,
            n_jobs=n_jobs,
            batch_size=batch_size,
            read_csv_params=read_csv_params,
        )

        if n_jobs == 1:
            res = [
                self.predict(df, features_names, return_all_predictions)
                for df in data_generator
            ]

        res = CupyDataset(
            cp.concatenate([x.data for x in res], axis=0),
            features=res[0].features,
            roles=res[0].roles,
        )
        if return_numpy:
            return res.to_numpy()
        else:
            return res


class TabularUtilizedAutoMLGPU(TabularUtilizedAutoML):
    """Template to make TimeUtilization from TabularAutoML (GPU version).
    Simplifies using ``TimeUtilization`` module for ``TabularAutoMLPreset``.

        Args:
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: CPU limit that that are passed to each automl.
            gpu_ids: GPU IDs that are passed to each automl.
            timing_params: Timing params level that are passed to each automl.
            configs_list: List of str path to configs files.
            drop_last: Usually last automl will be stopped with timeout.
              Flag that defines if we should drop it from ensemble.
            return_all_predictions: skip blending phase
            max_runs_per_config: Maximum number of multistart loops.
            random_state: Initial random seed that will be set
              in case of search in config.

    """

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = "all",
        timing_params: Optional[dict] = None,
        configs_list: Optional[Sequence[str]] = None,
        drop_last: bool = True,
        return_all_predictions: bool = False,
        max_runs_per_config: int = 5,
        random_state: int = 42,
        outer_blender_max_nonzero_coef: float = 0.05,
        **kwargs
    ):

        if configs_list is None:
            configs_list = [
                os.path.join(_base_dir, "tabular_configs_gpu", x)
                for x in [
                    "conf_0_sel_type_0.yml",
                    "conf_1_sel_type_1.yml",
                    "conf_2_select_mode_1_no_typ.yml",
                    "conf_3_sel_type_1_no_inter_lgbm.yml",
                    "conf_4_sel_type_0_no_int.yml",
                    "conf_5_sel_type_1_tuning_full.yml",
                    "conf_6_sel_type_1_tuning_full_no_int_lgbm.yml",
                ]
            ]
        inner_blend = MeanBlenderGPU()
        outer_blend = WeightedBlenderGPU(
            max_nonzero_coef=outer_blender_max_nonzero_coef
        )
        super(TabularUtilizedAutoMLGPU.__bases__[0], self).__init__(
            TabularAutoMLGPU,
            task,
            timeout,
            memory_limit,
            cpu_limit,
            gpu_ids,
            timing_params,
            configs_list,
            inner_blend,
            outer_blend,
            drop_last,
            return_all_predictions,
            max_runs_per_config,
            None,
            random_state,
            **kwargs
        )

    def predict(
        self,
        data: ReadableToDf,
        features_names: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        n_jobs: Optional[int] = 1,
        return_all_predictions: Optional[bool] = None,
    ) -> NumpyDataset:
        """Get dataset with predictions (GPU version).

        Additionally supports:
            - :class:`~cupy.ndarray`.
            - :class:`cudf.DataFrame`.
            - :class:`daskcudf.DataFrame`.

        Parallel inference is not available (n_jobs will be set to 1).

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
              if cannot be inferred from `train_data`.
            batch_size: Batch size or ``None``.
            n_jobs: Number of jobs (available only for consistency).
            return_all_predictions: if True,
              returns all model predictions from last level
        Returns:
            Dataset with predictions on CPU. Call ``.data`` to get predictions array.
        """
        return (
            super(TabularUtilizedAutoMLGPU.__bases__[0], self)
            .predict(
                data,
                features_names,
                return_all_predictions,
                batch_size=batch_size,
                n_jobs=n_jobs,
                return_numpy=False,
            )
            .to_numpy()
        )
