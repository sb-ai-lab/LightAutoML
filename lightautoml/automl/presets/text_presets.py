"""AutoML presets for data with texts."""

from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("nlp", error=True)


import logging
import os

from typing import Optional
from typing import Sequence

import torch

from pandas import DataFrame

from ...ml_algo.boost_cb import BoostCB
from ...ml_algo.boost_lgbm import BoostLGBM
from ...ml_algo.dl_model import TorchModel
from ...ml_algo.linear_sklearn import LinearLBFGS
from ...ml_algo.tuning.optuna import OptunaTuner
from ...pipelines.features.base import FeaturesPipeline
from ...pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from ...pipelines.features.linear_pipeline import LinearFeatures
from ...pipelines.features.text_pipeline import NLPTFiDFFeatures
from ...pipelines.features.text_pipeline import TextAutoFeatures
from ...pipelines.features.text_pipeline import TextBertFeatures
from ...pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline
from ...pipelines.selection.base import SelectionPipeline
from ...reader.base import PandasToPandasReader
from ...reader.tabular_batch_generator import ReadableToDf
from ...tasks import Task
from ..blend import WeightedBlender
from .base import upd_params
from .tabular_presets import NumpyDataset
from .tabular_presets import TabularAutoML


logger = logging.getLogger(__name__)


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
class TabularNLPAutoML(TabularAutoML):
    """Classic preset - work with tabular and text data.

    Supported data roles - numbers, dates, categories, text
    Limitations - no memory management.

    GPU support in catboost/lightgbm (if installed for GPU),
    NN models training.
    """

    _default_config_path = "text_config.yml"

    _time_scores = {
        "lgb": 1,
        "lgb_tuned": 3,
        "linear_l2": 0.7,
        "cb": 2,
        "cb_tuned": 6,
        "nn": 1,
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
        nn_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None,
        cb_params: Optional[dict] = None,
        linear_l2_params: Optional[dict] = None,
        nn_pipeline_params: Optional[dict] = None,
        gbm_pipeline_params: Optional[dict] = None,
        linear_pipeline_params: Optional[dict] = None,
        text_params: Optional[dict] = None,
        tfidf_params: Optional[dict] = None,
        autonlp_params: Optional[dict] = None,
    ):

        """

        Commonly _params kwargs (ex. timing_params) set via
        config file (config_path argument).
        If you need to change just few params, it's possible to pass
        it as dict of dicts, like json.
        To get available params please look on default config template.
        Also you can find there param description.
        To generate config template call
        :meth:`TabularNLPAutoML.get_config('config_path.yml')`.


        Args:
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: CPU limit that that are passed to each automl.
            gpu_ids: GPU IDs that are passed to each automl.
            timing_params: Timing param dict.
            config_path: Path to config file.
            general_params: General param dict.
            reader_params: Reader param dict.
            read_csv_params: Params to pass :func:`pandas.read_csv`
              (case of train/predict from file).
            nested_cv_params: Param dict for nested cross-validation.
            tuning_params: Params of Optuna tuner.
            selection_params: Params of feature selection.
            nn_params: Params of neural network model.
            lgb_params: Params of lightgbm model.
            cb_params: Params of catboost model.
            linear_l2_params: Params of linear model.
            nn_pipeline_params: Params of feature generation
              for neural network models.
            gbm_pipeline_params: Params of feature generation
              for boosting models.
            linear_pipeline_params: Params of feature generation
              for linear models.
            text_params: General params of text features.
            tfidf_params: Params of tfidf features.
            autonlp_params: Params of text embeddings features.

        """
        super().__init__(
            task,
            timeout,
            memory_limit,
            cpu_limit,
            gpu_ids,
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
                "nn_params",
                "lgb_params",
                "cb_params",
                "linear_l2_params",
                "nn_pipeline_params",
                "gbm_pipeline_params",
                "linear_pipeline_params",
                "text_params",
                "tfidf_params",
                "autonlp_params",
            ],
            [
                general_params,
                reader_params,
                read_csv_params,
                nested_cv_params,
                tuning_params,
                selection_params,
                nn_params,
                lgb_params,
                cb_params,
                linear_l2_params,
                nn_pipeline_params,
                gbm_pipeline_params,
                linear_pipeline_params,
                text_params,
                tfidf_params,
                autonlp_params,
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

            self.nn_params["device"] = gpu_ids.split(",")
            self.text_params["device"] = gpu_ids.split(",")

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "cb", "nn"]]

        else:
            self.nn_params["device"] = "cpu"
            self.text_params["device"] = "cpu"

            if self.general_params["use_algos"] == "auto":
                self.general_params["use_algos"] = [["linear_l2", "lgb"]]

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        torch.set_num_threads(cpu_cnt)

        self.nn_params["num_workers"] = min(self.nn_params["num_workers"], cpu_cnt)
        self.nn_params["lang"] = self.nn_params["lang"] or self.text_params["lang"]
        self.nn_params["bert_name"] = self.nn_params["bert_name"] or self.text_params["bert_model"]

        logger.info3("Model language mode: {}".format(self.nn_params["lang"]))

        if isinstance(self.autonlp_params["transformer_params"], dict):
            if "loader_params" in self.autonlp_params["transformer_params"]:
                self.autonlp_params["transformer_params"]["loader_params"]["num_workers"] = min(
                    self.autonlp_params["transformer_params"]["loader_params"]["num_workers"],
                    cpu_cnt,
                )
            else:
                self.autonlp_params["transformer_params"]["loader_params"] = {"num_workers": cpu_cnt}

        # other params as tabular
        super().infer_auto_params(train_data, multilevel_avail)

    def get_nlp_pipe(self, type: str = "tfidf") -> Optional[FeaturesPipeline]:
        if type == "tfidf":
            return NLPTFiDFFeatures(**self.text_params, **self.tfidf_params)
        elif type == "embed":
            return TextAutoFeatures(**self.text_params, **self.autonlp_params)
        elif type == "bert":
            return TextBertFeatures(**self.text_params)
        else:
            return None

    def get_nn(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None) -> NestedTabularMLPipeline:

        # nn model
        time_score = self.get_time_score(n_level, "nn")
        nn_timer = self.timer.get_task_timer("reg_nn", time_score)
        nn_model = TorchModel(timer=nn_timer, default_params=self.nn_params)

        text_nn_feats = self.get_nlp_pipe(self.nn_pipeline_params["text_features"])
        nn_feats = LinearFeatures(output_categories=True, **self.linear_pipeline_params)
        if text_nn_feats is not None:
            nn_feats.append(text_nn_feats)

        nn_pipe = NestedTabularMLPipeline(
            [nn_model], pre_selection=None, features_pipeline=nn_feats, **self.nested_cv_params
        )
        return nn_pipe

    def get_linear(self, n_level: int = 1, pre_selector: Optional[SelectionPipeline] = None) -> NestedTabularMLPipeline:

        # linear model with l2
        time_score = self.get_time_score(n_level, "linear_l2")
        linear_l2_timer = self.timer.get_task_timer("reg_l2", time_score)
        linear_l2_model = LinearLBFGS(timer=linear_l2_timer, **self.linear_l2_params)

        text_l2_feats = self.get_nlp_pipe(self.linear_pipeline_params["text_features"])
        linear_l2_feats = LinearFeatures(output_categories=True, **self.linear_pipeline_params)
        if text_l2_feats is not None:
            linear_l2_feats.append(text_l2_feats)

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

        text_gbm_feats = self.get_nlp_pipe(self.gbm_pipeline_params["text_features"])
        gbm_feats = LGBAdvancedPipeline(output_categories=False, **self.gbm_pipeline_params)
        if text_gbm_feats is not None:
            gbm_feats.append(text_gbm_feats)

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

        Args:
            **fit_args: Contain all information needed for creating automl.

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

            if "nn" in names:
                selector = None
                lvl.append(self.get_nn(n + 1, selector))

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

        Returns:
            Dataset with predictions.

        """
        return super().predict(data, features_names, batch_size)
