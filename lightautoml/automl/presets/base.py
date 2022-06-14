"""AutoML presets base class."""

import logging
import os
import shutil

from typing import Any
from typing import Iterable
from typing import Optional
from typing import Sequence

import torch
import yaml

from ...dataset.base import LAMLDataset
from ...tasks import Task
from ...utils.logging import add_filehandler
from ...utils.logging import set_stdout_level
from ...utils.logging import verbosity_to_loglevel
from ...utils.timer import PipelineTimer
from ..base import AutoML


logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)


def upd_params(old: dict, new: dict) -> dict:
    for k in new:
        if type(new[k]) is dict and k in old and type(old[k]) is dict:
            upd_params(old[k], new[k])
        else:
            old[k] = new[k]

    return old


class AutoMLPreset(AutoML):
    """Basic class for automl preset.

    It's almost like AutoML, but with delayed initialization.
    Initialization starts on fit, some params are inferred from data.
    Preset should be defined via ``.create_automl`` method.
    Params should be set via yaml config.
    Most usefull case - end-to-end model development.

    Example:

        >>> automl = SomePreset(Task('binary'), timeout=3600)
        >>> automl.fit_predict(data, roles={'target': 'TARGET'})

    """

    _default_config_path = "example_config.yml"

    def __init__(
        self,
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = "all",
        timing_params: Optional[dict] = None,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """

        Commonly _params kwargs (ex. timing_params) set via
        config file (config_path argument).
        If you need to change just few params,
        it's possible to pass it as dict of dicts, like json.
        To get available params please look on default config template.
        Also you can find there param description.
        To generate config template
        call ``SomePreset.get_config('config_path.yml')``.

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
            **kwargs: Not used.

        """
        self._set_config(config_path)

        for name, param in zip(["timing_params"], [timing_params]):
            if param is None:
                param = {}
            self.__dict__[name] = {**self.__dict__[name], **param}

        self.timer = PipelineTimer(timeout, **getattr(self, "timing_params"))
        self.memory_limit = memory_limit
        if cpu_limit == -1:
            cpu_limit = os.cpu_count()
        self.cpu_limit = cpu_limit
        self.gpu_ids = gpu_ids
        if gpu_ids == "all":
            self.gpu_ids = ",".join(map(str, range(torch.cuda.device_count())))
        self.task = task

    def _set_config(self, path):
        self.config_path = path

        if path is None:
            path = os.path.join(base_dir, self._default_config_path)

        with open(path) as f:
            params = yaml.safe_load(f)

        for k in params:
            self.__dict__[k] = params[k]

    @classmethod
    def get_config(cls, path: Optional[str] = None) -> Optional[dict]:
        """Create new config template.

        Args:
            path: Path to config.

        Returns:
            Config.

        """
        if path is None:
            path = os.path.join(base_dir, cls._default_config_path)
            with open(path) as f:
                params = yaml.safe_load(f)
            return params

        else:
            shutil.copy(os.path.join(base_dir, cls._default_config_path), path)

    def create_automl(self, **fit_args):
        """Abstract method - how to build automl.

        Here you should create all automl components,
        like readers, levels, timers, blenders.
        Method ``._initialize`` should be called in the end to create automl.

        Args:
            **fit_args: params that are passed to ``.fit_predict`` method.

        """
        raise NotImplementedError

    def fit_predict(
        self,
        train_data: Any,
        roles: dict,
        train_features: Optional[Sequence[str]] = None,
        cv_iter: Optional[Iterable] = None,
        valid_data: Optional[Any] = None,
        valid_features: Optional[Sequence[str]] = None,
        verbose: int = 0,
    ) -> LAMLDataset:
        """Fit on input data and make prediction on validation part.

        Args:
            train_data: Dataset to train.
            roles: Roles dict.
            train_features: Features names,
              if can't be inferred from `train_data`.
            cv_iter: Custom cv-iterator. For example,
              :class:`~lightautoml.validation.np_iterators.TimeSeriesIterator`.
            valid_data: Optional validation dataset.
            valid_features: Optional validation dataset features if can't be
              inferred from `valid_data`.
            verbose: Verbosity level that are passed to each automl.

        Returns:
            Dataset with predictions. Call ``.data`` to get predictions array.

        """
        self.set_verbosity_level(verbose)

        self.create_automl(
            train_data=train_data,
            roles=roles,
            train_features=train_features,
            cv_iter=cv_iter,
            valid_data=valid_data,
            valid_features=valid_features,
        )
        logger.info(f"Task: {self.task.name}\n")

        logger.info("Start automl preset with listed constraints:")
        logger.info(f"- time: {self.timer.timeout:.2f} seconds")
        logger.info(f"- CPU: {self.cpu_limit} cores")
        logger.info(f"- memory: {self.memory_limit} GB\n")

        self.timer.start()
        result = super().fit_predict(
            train_data,
            roles,
            train_features,
            cv_iter,
            valid_data,
            valid_features,
            verbose,
        )

        logger.info("\x1b[1mAutoml preset training completed in {:.2f} seconds\x1b[0m\n".format(self.timer.time_spent))
        logger.info(f"Model description:\n{self.create_model_str_desc()}\n")

        return result

    def create_model_str_desc(self, pref_tab_num: int = 0, split_line_len: int = 0) -> str:
        prefix = "\t" * pref_tab_num
        splitter = prefix + "=" * split_line_len + "\n"
        model_stats = sorted(list(self.collect_model_stats().items()))

        last_lvl = model_stats[-1][0].split("_")[1]
        last_lvl_models = [ms for ms in model_stats if ms[0].startswith("Lvl_" + last_lvl)]
        notlast_lvl_models = [ms for ms in model_stats if not ms[0].startswith("Lvl_" + last_lvl)]

        res = ""
        if len(notlast_lvl_models) > 0:
            cur_level = 0
            res += prefix + "Models on level 0:\n"
            for model_stat in notlast_lvl_models:
                model_name, cnt_folds = model_stat
                level = int(model_name.split("_")[1])
                if level != cur_level:
                    cur_level = level
                    res += "\n" + prefix + "Models on level {}:\n".format(cur_level)
                res += prefix + "\t {} averaged models {}\n".format(cnt_folds, model_name)
            res += "\n"

        res += prefix + "Final prediction for new objects (level {}) = \n".format(last_lvl)
        for model_stat, weight in zip(last_lvl_models, self.blender.wts):
            model_name, cnt_folds = model_stat
            res += prefix + "\t {:.5f} * ({} averaged models {}) +\n".format(weight, cnt_folds, model_name)

        if split_line_len == 0:
            return res[:-2]

        return splitter + res[:-2] + "\n" + splitter

    @staticmethod
    def set_verbosity_level(verbose: int):
        """Verbosity level setter.

        Args:
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;

        """
        level = verbosity_to_loglevel(verbose)
        set_stdout_level(level)

        logger.info(f"Stdout logging level is {logging._levelToName[level]}.")

    @staticmethod
    def set_logfile(filename: str):
        """"""
        add_filehandler(filename)
