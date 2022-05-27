"""Whitebox AutoML presets."""

import logging
import os

from copy import deepcopy
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import cast

from ...dataset.np_pd_dataset import NumpyDataset
from ...ml_algo.whitebox import WbMLAlgo
from ...pipelines.ml.whitebox_ml_pipe import WBPipeline
from ...reader.base import PandasToPandasReader
from ...tasks import Task
from .base import AutoMLPreset
from .base import upd_params


logger = logging.getLogger(__name__)
_base_dir = os.path.dirname(__file__)


class WhiteBoxPreset(AutoMLPreset):
    """Preset for AutoWoE - logistic regression over binned features (scorecard).

    Supported data roles - numbers, dates, categories.

    Limitations:

        - Simple time management.
        - No memory management.
        - Working only with :class:`pandas.DataFrame`.
        - No batch inference.
        - No text support.
        - No parallel execution.
        - No batch inference.
        - No GPU usage.
        - No cross-validation scheme. Supports only holdout validation
          (cv is created inside AutoWoE, but no oof pred returned).

    Common usecase - fit lightweight interpretable model
    for binary classification task.

    """

    _default_config_path = "whitebox_config.yml"

    @property
    def whitebox(self):
        """Get wrapped AutoWoE object.

        Returns:
            Model.

        """
        return self.levels[0][0].whitebox

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = None,
        verbose: int = 2,
        timing_params: Optional[dict] = None,
        config_path: Optional[str] = None,
        general_params: Optional[dict] = None,
        reader_params: Optional[dict] = None,
        read_csv_params: Optional[dict] = None,
        whitebox_params: Optional[dict] = None,
    ):
        """

        Commonly _params kwargs (ex. timing_params) set via
        config file (config_path argument).
        If you need to change just few params, it's possible to pass it
        as dict of dicts, like json.
        To get available params please look on default config template.
        Also you can find there param description
        To generate config template call
        :func:`WhiteBoxPreset.get_config('config_path.yml')`.

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
            general_params: General param dict.
            reader_params: Reader param dict.
            read_csv_params: Params to pass :func:`pandas.read_csv`
              (case of train/predict from file).
            whitebox_params: Params of WhiteBox algo (look at config file).

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
                "whitebox_params",
            ],
            [
                general_params,
                reader_params,
                read_csv_params,
                whitebox_params,
            ],
        ):
            if param is None:
                param = {}
            self.__dict__[name] = upd_params(self.__dict__[name], param)

    def infer_auto_params(self, **kwargs):

        # check all n_jobs params
        cpu_cnt = min(os.cpu_count(), self.cpu_limit)
        self.whitebox_params["default_params"]["n_jobs"] = min(
            self.whitebox_params["default_params"]["n_jobs"], cpu_cnt
        )
        self.reader_params["n_jobs"] = min(self.reader_params["n_jobs"], cpu_cnt)
        self.whitebox_params["verbose"] = self.verbose

    def create_automl(self, *args, **kwargs):
        """Create basic :class:`~lightautoml.automl.presets.whitebox_presets.WhiteBoxPreset` instance from data.

        Args:
            *args: Not used.
            **kwargs: everything passed to ``.fit_predict``.

        """
        self.infer_auto_params()
        reader = PandasToPandasReader(task=self.task, **self.reader_params)
        wb_timer = self.timer.get_task_timer("wb", 1)

        whitebox_params = deepcopy(self.whitebox_params)
        whitebox_params["fit_params"] = self.fit_params
        whitebox_params["report"] = self.general_params["report"]

        _whitebox = WbMLAlgo(timer=wb_timer, default_params=whitebox_params)
        whitebox = WBPipeline(_whitebox)
        levels = [[whitebox]]

        # initialize
        self._initialize(reader, levels, skip_conn=False, timer=self.timer, verbose=self.verbose)

    def fit_predict(
        self,
        train_data: Any,
        roles: dict,
        train_features: Optional[Sequence[str]] = None,
        cv_iter: Optional[Iterable] = None,
        valid_data: Optional[Any] = None,
        valid_features: Optional[Sequence[str]] = None,
        **fit_params
    ) -> NumpyDataset:
        """Fit and get prediction on validation dataset.

        Almost same as :meth:`lightautoml.automl.base.AutoML.fit_predict`.

        Additional features - working with different data formats.
        Supported now:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
              For example, ``{'data': X...}``. In this case,
              roles are optional, but `train_features`
              and `valid_features` required.
            - :class:`pandas.DataFrame`.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Optional features names, if can't
              be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example,
              :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features
              if cannot be inferred from `valid_data`.

        Returns:
            Dataset with predictions. Call ``.data`` to get predictions array.

        """
        assert cv_iter is None or len(cv_iter) == 2, "Expect custom iterator with len 2"
        if valid_data is None and cv_iter is None:
            logger.info2("Validation data is not set. Train will be used as valid in report and valid prediction")
            valid_data = train_data
            valid_features = train_features

        self.fit_params = fit_params
        pred = super().fit_predict(train_data, roles, train_features, cv_iter, valid_data, valid_features)
        return cast(NumpyDataset, pred)

    def predict(
        self,
        data: Any,
        features_names: Optional[Sequence[str]] = None,
        report: bool = False,
    ) -> NumpyDataset:
        """Almost same as AutoML ``.predict`` with additional features.

        Additional features - generate extended WhiteBox
        ``report=True`` passed to args.

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
              if can't be inferred from `train_data`.
            report: Flag if we need inner WhiteBox report update
              (``True`` is slow). Only if ``general_params['report'] = True``.

        Returns:
            Dataset with predictions.

        """

        dataset = self.reader.read(data, features_names, add_array_attrs=report)
        pred = self.levels[0][0].predict(dataset, report=report)

        return cast(NumpyDataset, pred)
