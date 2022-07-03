"""AutoML presets for image data."""

from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("cv", error=True)


import os

from typing import Optional
from typing import Sequence

import torch

from pandas import DataFrame

from ...ml_algo.boost_cb import BoostCB
from ...ml_algo.boost_lgbm import BoostLGBM
from ...ml_algo.linear_sklearn import LinearLBFGS
from ...ml_algo.tuning.optuna import OptunaTuner
from ...pipelines.features.base import FeaturesPipeline
from ...pipelines.features.image_pipeline import ImageAutoFeatures
from ...pipelines.features.image_pipeline import ImageSimpleFeatures
from ...pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from ...pipelines.features.linear_pipeline import LinearFeatures
from ...pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline
from ...pipelines.selection.base import SelectionPipeline
from ...reader.base import PandasToPandasReader
from ...reader.tabular_batch_generator import ReadableToDf
from ...tasks import Task
from ..blend import WeightedBlender
from .base import upd_params
from .tabular_presets import NumpyDataset
from .tabular_presets import TabularAutoML


_base_dir = os.path.dirname(__file__)
# set initial runtime rate guess for first level models
_time_scores = {
    "lgb": 1,
    "lgb_tuned": 3,
    "linear_l2": 0.7,
    "cb": 2,
    "cb_tuned": 6,
    "nn": 1,
}


# TODO: add text feature selection
class TabularCVAutoML(TabularAutoML):
    """Classic preset - work with tabular and image data.

    Supported data roles - numbers, dates, categories, images.
    Limitations - no memory management.
    GPU support in catboost/lightgbm (if installed for GPU).

    """

    _default_config_path = "image_config.yml"

    _time_scores = {
        "lgb": 1,
        "lgb_tuned": 3,
        "linear_l2": 0.7,
        "cb": 2,
        "cb_tuned": 6,
    }

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = "all",
        verbose: int = 2,
        timing_params: Optional[dict] = None,
        config_path: Optional[str] = None,
        general_params: Optional[dict] = None,
        reader_params: Optional[dict] = None,
        read_csv_params: Optional[dict] = None,
        nested_cv_params: Optional[dict] = None,
        tuning_params: Optional[dict] = None,
        selection_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        cb_params: Optional[dict] = None,
        linear_l2_params: Optional[dict] = None,
        gbm_pipeline_params: Optional[dict] = None,
        linear_pipeline_params: Optional[dict] = None,
        cv_simple_features: Optional[dict] = None,
        autocv_features: Optional[dict] = None,
    ):

        """

        Commonly _params kwargs (ex. timing_params)
        set via config file (config_path argument).
        If you need to change just few params,
        it's possible to pass it as dict of dicts, like json.
        To get available params please look on default config template.
        Also you can find there param description.
        To generate config template call
        :func:`TabularCVAutoML.get_config('config_path.yml')`.

        Args:
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: CPU limit that that are passed to each automl.
            gpu_ids: GPU IDs that are passed to each automl.
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;
            timing_params: Timing param dict.
            config_path: Path to config file.
            general_params: General param dict
            reader_params: Reader param dict.
            read_csv_params: Params to pass :func:`pandas.read_csv`
              (case of train/predict from file).
            nested_cv_params: Param dict for nested cross-validation.
            tuning_params: Params of Optuna tuner.
            selection_params: Params of feature selection.
            lgb_params: Params of lightgbm model.
            cb_params: Params of catboost model.
            linear_l2_params: Params of linear model.
            gbm_pipeline_params: Params of feature generation
              for boosting models.
            linear_pipeline_params: Params of feature generation
              for linear models.
            cv_simple_features: Params of color histogram features.
            autocv_features: Params of image embeddings features.

        """
        super().__init__(
            task,
            timeout,
            memory_limit,
            cpu_limit,
            gpu_ids,
            verbose,
            timing_params,
            config_path,
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
                "lgb_params",
                "cb_params",
                "linear_l2_params",
                "gbm_pipeline_params",
                "linear_pipeline_params",
                "cv_simple_features",
                "autocv_features",
            ],
            [
                general_params,
                reader_params,
                read_csv_params,
                nested_cv_params,
                tuning_params,
                selection_params,
                lgb_params,
                cb_params,
                linear_l2_params,
                gbm_pipeline_params,
                linear_pipeline_params,
                cv_simple_features,
                autocv_features,
            ],
        ):
            if param is None:
                param = {}
            self.__dict__[name] = upd_params(self.__dict__[name], param)

    def infer_auto_params(self, train_data: DataFrame, multilevel_avail: bool = False):

        # infer gpu params
        gpu_cnt = torch.cuda.device_count()
        gpu_ids = self.gpu_ids
        if gpu_cnt > 0 and gpu_ids:
            if gpu_ids == "all":
                gpu_ids = ",".join(list(map(str, range(gpu_cnt))))

            self.autocv_features["device"] = gpu_ids.split(",")

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "cb"]]

        else:
            self.autocv_features["device"] = "cpu"

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "lgb"]]

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        if "n_jobs" in self.autocv_features:
            self.autocv_features["n_jobs"] = min(self.autocv_features["n_jobs"], cpu_cnt)
        else:
            self.autocv_features["n_jobs"] = cpu_cnt

        if "n_jobs" in self.cv_simple_features:
            self.cv_simple_features["n_jobs"] = min(self.cv_simple_features["n_jobs"], cpu_cnt)
        else:
            self.cv_simple_features["n_jobs"] = cpu_cnt

        # other params as tabular
        super().infer_auto_params(train_data, multilevel_avail)

    def get_cv_pipe(self, type: str = "simple") -> Optional[FeaturesPipeline]:
        if type == "simple":
            return ImageSimpleFeatures(**self.cv_simple_features)
        elif type == "embed":
            return ImageAutoFeatures(**self.autocv_features)
        else:
            return None

    def get_linear(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None) -> NestedTabularMLPipeline:

        # linear model with l2
        time_score = self.get_time_score(n_level, "linear_l2")
        linear_l2_timer = self.timer.get_task_timer("reg_l2", time_score)
        linear_l2_model = LinearLBFGS(timer=linear_l2_timer, **self.linear_l2_params)

        cv_l2_feats = self.get_cv_pipe(self.linear_pipeline_params["cv_features"])
        linear_l2_feats = LinearFeatures(output_categories=True, **self.linear_pipeline_params)
        if cv_l2_feats is not None:
            linear_l2_feats.append(cv_l2_feats)

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
    ):

        cv_gbm_feats = self.get_cv_pipe(self.gbm_pipeline_params["cv_features"])
        gbm_feats = LGBAdvancedPipeline(output_categories=False, **self.gbm_pipeline_params)
        if cv_gbm_feats is not None:
            gbm_feats.append(cv_gbm_feats)

        ml_algos = []
        force_calc = []
        for key, force in zip(keys, [True, False, False, False]):
            tuned = "_tuned" in key
            algo_key = key.split("_")[0]
            time_score = self.get_time_score(n_level, key)
            gbm_timer = self.timer.get_task_timer(algo_key, time_score)
            if algo_key == "lgb":
                gbm_model = BoostLGBM(timer=gbm_timer, **self.lgb_params)
            elif algo_key == "cb":
                gbm_model = BoostCB(timer=gbm_timer, **self.cb_params)
            else:
                raise ValueError("Wrong algo key")

            if tuned:
                gbm_tuner = OptunaTuner(
                    n_trials=self.tuning_params["max_tuning_iter"],
                    timeout=self.tuning_params["max_tuning_time"],
                    fit_on_holdout=self.tuning_params["fit_on_holdout"],
                )
                gbm_model = (gbm_model, gbm_tuner)
            ml_algos.append(gbm_model)
            force_calc.append(force)

        gbm_pipe = NestedTabularMLPipeline(
            ml_algos, force_calc, pre_selection=pre_selector, features_pipeline=gbm_feats, **self.nested_cv_params
        )

        return gbm_pipe

    def create_automl(self, **fit_args):
        """Create basic automl instance.

        See :meth:`lightautoml.automl.presets.base.AutoMLPreset.create_automl`.

        Args:
            **fit_args: params that are passed to ``.fit_predict`` method.

        """

        train_data = fit_args["train_data"]
        self.infer_auto_params(train_data)
        reader = PandasToPandasReader(task=self.task, **self.reader_params)

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
                x for x in ["lgb", "lgb_tuned", "cb", "cb_tuned"] if x in names and x.split("_")[0] in self.task.losses
            ]

            if len(gbm_models) > 0:
                selector = None
                if "gbm" in self.selection_params["select_algos"] and (self.general_params["skip_conn"] or n == 0):
                    selector = pre_selector
                lvl.append(self.get_gbms(gbm_models, n + 1, selector))

            levels.append(lvl)

        # blend everything
        blender = WeightedBlender()

        # initialize
        self._initialize(
            reader,
            levels,
            skip_conn=self.general_params["skip_conn"],
            blender=blender,
            timer=self.timer,
            verbose=self.verbose,
        )

    def predict(
        self,
        data: ReadableToDf,
        features_names: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        n_jobs: Optional[int] = 1,
    ) -> NumpyDataset:
        """Get dataset with predictions.

        Almost same as :meth:`lightautoml.automl.base.AutoML.predict`
        on new dataset, with additional features.

        Additional features - working with different data formats.
        Supported now:

            - Path to `.csv`, `.parquet`, `.feather` files.
            - ``np.ndarray``, or dict of ``np.ndarray``. For example,
              ``{'data': X...}``. In this case roles are optional,
              but `train_features` and `valid_features` required.
            - ``pandas.DataFrame``.

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

        Returns:
            Dataset with predictions.

        """
        return super().predict(data, features_names, batch_size)
