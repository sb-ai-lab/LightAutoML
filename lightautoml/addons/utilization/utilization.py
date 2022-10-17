"""Tools to configure time utilization."""

import logging

from copy import deepcopy
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

from ...automl.base import AutoML
from ...automl.blend import BestModelSelector
from ...automl.blend import Blender
from ...automl.presets.base import AutoMLPreset
from ...dataset.base import LAMLDataset
from ...dataset.utils import concatenate
from ...ml_algo.base import MLAlgo
from ...pipelines.ml.base import MLPipeline
from ...tasks import Task
from ...utils.logging import set_stdout_level
from ...utils.logging import verbosity_to_loglevel
from ...utils.timer import PipelineTimer


logger = logging.getLogger(__name__)


class MLAlgoForAutoMLWrapper(MLAlgo):
    """Wrapper to apply blender to list of automl's."""

    @classmethod
    def from_automls(cls, automl: Union[AutoML, Sequence[AutoML]]):
        """Constructs automls.

        Args:
            automl: One AutoML or list of AutoML objects.

        Returns:
            MLAlgo.

        """
        ml_algo = cls()
        ml_algo.models.append(automl)

        return ml_algo

    def fit_predict(self, *args, **kwargs) -> LAMLDataset:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> LAMLDataset:
        raise NotImplementedError


class MLPipeForAutoMLWrapper(MLPipeline):
    """Wrapper to apply blender to list of automls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_algos = self._ml_algos

    @classmethod
    def from_automl(cls, automl: AutoML):
        ml_pipe = cls([MLAlgoForAutoMLWrapper.from_automls(automl)])

        return ml_pipe

    @classmethod
    def from_blended(cls, automls: Sequence[AutoML], blender: Blender):
        ml_pipe = cls(
            [
                MLAlgoForAutoMLWrapper.from_automls(automls),
            ]
        )
        ml_pipe.blender = blender

        return ml_pipe


class TimeUtilization:
    """Class that helps to utilize given time to :class:`~lightautoml.automl.presets.base.AutoMLPreset`.

    Useful to calc benchmarks and compete
    It takes list of config files as input and run it white time limit exceeded.
    If time left - it can perform multistart on same configs with new random state.
    In best case - blend different configurations of single preset.
    In worst case - averaging multiple automl's with different states.

    Note:
        Basic usage.

        >>> ensembled_automl = TimeUtilization(TabularAutoML, Task('binary'),
        >>>     timeout=3600, configs_list=['cfg0.yml', 'cfg1.yml'])

        Then ``.fit_predict`` and predict can be
        called like usual :class:`~lightautoml.automl.base.AutoML` class.

    """

    def __init__(
        self,
        automl_factory: Type[AutoMLPreset],
        task: Task,
        timeout: int = 3600,
        memory_limit: int = 16,
        cpu_limit: int = 4,
        gpu_ids: Optional[str] = None,
        timing_params: Optional[dict] = None,
        configs_list: Optional[Sequence[str]] = None,
        inner_blend: Optional[Blender] = None,
        outer_blend: Optional[Blender] = None,
        drop_last: bool = True,
        return_all_predictions: bool = False,
        max_runs_per_config: int = 5,
        random_state_keys: Optional[dict] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """

        Args:
            automl_factory: One of presets.
            task: Task to solve.
            timeout: Timeout in seconds.
            memory_limit: Memory limit that are passed to each automl.
            cpu_limit: Cpu limit that that are passed to each automl.
            gpu_ids: Gpu_ids that are passed to each automl.
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;
            timing_params: Timing_params level that are passed to each automl.
            configs_list: List of str path to configs files.
            inner_blend: Blender instance to blend automl's with same configs
              and different random state.
            outer_blend: Blender instance to blend averaged by random_state
              automl's with different configs.
            drop_last: Usually last automl will be stopped with timeout.
              Flag that defines if we should drop it from ensemble
            return_all_predictions: Skip blend and return all model predictions
            max_runs_per_config: Maximum number of multistart loops.
            random_state_keys: Params of config that used as
              random state with initial values. If ``None`` - search for
              `random_state` key in default config of preset.
              If not found - assume, that seeds are not fixed
              and each run is random by default. For example
              ``{'reader_params': {'random_state': 42}, 'gbm_params': {'default_params': {'seed': 42}}}``
            random_state: initial random seed, that will be
              set in case of search in config.
            **kwargs: Additional params.

        """

        self.automl_factory = automl_factory
        self.task = task
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.gpu_ids = gpu_ids

        self.timing_params = timing_params
        if timing_params is None:
            self.timing_params = {}

        self.configs_list = configs_list
        if configs_list is None:
            self.configs_list = [None]

        self.max_runs_per_config = max_runs_per_config

        self.random_state_keys = random_state_keys
        if random_state_keys is None:
            self.random_state_keys = self._search_for_states(automl_factory, random_state)

        self.inner_blend = inner_blend
        if inner_blend is None:
            self.inner_blend = BestModelSelector()

        self.outer_blend = outer_blend
        if outer_blend is None:
            self.outer_blend = BestModelSelector()
        self.drop_last = drop_last
        self.return_all_predictions = return_all_predictions
        self.kwargs = kwargs

    def _search_for_key(self, config, key, value: int = 42) -> dict:

        d = {}

        if key in config:
            d[key] = value

        for k in config:
            if type(config[k]) is dict:
                s = self._search_for_key(config[k], key, value)
                if len(s) > 0:
                    d[k] = s
        return d

    def _search_for_states(self, automl_factory: Type[AutoMLPreset], random_state: int = 42) -> dict:

        config = automl_factory.get_config()
        random_states = self._search_for_key(config, "random_state", random_state)

        return random_states

    def _get_upd_states(self, random_state_keys: dict, upd_value: int = 0) -> dict:

        d = {}

        for k in random_state_keys:
            if type(random_state_keys[k]) is dict:
                d[k] = self._get_upd_states(random_state_keys[k], upd_value)
            else:
                d[k] = random_state_keys[k] + upd_value

        return d

    def fit_predict(
        self,
        train_data: Any,
        roles: dict,
        train_features: Optional[Sequence[str]] = None,
        cv_iter: Optional[Iterable] = None,
        valid_data: Optional[Any] = None,
        valid_features: Optional[Sequence[str]] = None,
        verbose: int = 0,
        log_file: str = None,
    ) -> LAMLDataset:
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
        set_stdout_level(verbosity_to_loglevel(verbose))

        logger.info("Start automl \x1b[1mutilizator\x1b[0m with listed constraints:")
        logger.info(f"- time: {self.timeout:.2f} seconds")
        logger.info(f"- CPU: {self.cpu_limit} cores")
        logger.info(f"- memory: {self.memory_limit} GB\n")
        logger.info("\x1b[1mIf one preset completes earlier, next preset configuration will be started\x1b[0m\n")

        timer = PipelineTimer(self.timeout, **self.timing_params).start()
        history = []

        amls = [[] for _ in range(len(self.configs_list))]
        aml_preds = [[] for _ in range(len(self.configs_list))]
        n_ms = 0
        n_cfg = 0
        upd_state_val = 0
        flg_continute = True
        # train automls one by one while timer is ok
        while flg_continute:
            n_ms += 1

            logger.info("=" * 50)

            for n_cfg, config in enumerate(self.configs_list):
                random_states = self._get_upd_states(self.random_state_keys, upd_state_val)
                random_states["general_params"] = {"return_all_predictions": False}
                upd_state_val += 1

                logger.info(f"Start {n_cfg} automl preset configuration:")
                logger.info("\x1b[1m{}\x1b[0m, random state: {}".format(config.split("/")[-1], random_states))

                cur_kwargs = self.kwargs.copy()
                for k in random_states.keys():
                    if k in self.kwargs:
                        logger.info3("Found {} in kwargs, need to combine".format(k))
                        random_states[k] = {**cur_kwargs[k], **random_states[k]}
                        del cur_kwargs[k]
                        logger.info3("Merged variant for {} = {}".format(k, random_states[k]))

                automl = self.automl_factory(
                    self.task,
                    timer.time_left,
                    memory_limit=self.memory_limit,
                    cpu_limit=self.cpu_limit,
                    gpu_ids=self.gpu_ids,
                    timing_params=self.timing_params,
                    config_path=config,
                    **random_states,
                    **cur_kwargs,
                )

                val_pred = automl.fit_predict(
                    train_data,
                    roles,
                    train_features,
                    cv_iter,
                    valid_data,
                    valid_features,
                    verbose=verbose,
                    log_file=log_file,
                )

                logger.info("=" * 50)

                amls[n_cfg].append(MLPipeForAutoMLWrapper.from_automl(automl))
                aml_preds[n_cfg].append(val_pred)

                history.append(timer.time_spent - sum(history))
                if timer.time_left < (sum(history) / len(history)) or upd_state_val >= (
                    self.max_runs_per_config * len(self.configs_list)
                ):
                    flg_continute = False
                    break

        # usually last model will be not complete due to timeout.
        # Maybe it's better to remove it from inner blend, which is typically just mean of models
        if n_ms > 1 and self.drop_last:
            amls[n_cfg].pop()
            aml_preds[n_cfg].pop()

        # prune empty algos
        amls = [x for x in amls if len(x) > 0]
        aml_preds = [x for x in aml_preds if len(x) > 0]

        # blend - first is inner blend - we blend same config with different states
        inner_pipes = []
        inner_preds = []

        for preds, pipes in zip(aml_preds, amls):
            inner_blend = deepcopy(self.inner_blend)
            val_pred, inner_pipe = inner_blend.fit_predict(preds, pipes)
            inner_pipe = [x.ml_algos[0].models[0] for x in inner_pipe]

            inner_preds.append(val_pred)
            inner_pipes.append(MLPipeForAutoMLWrapper.from_blended(inner_pipe, inner_blend))

        # outer blend - blend of blends
        if not self.return_all_predictions:
            val_pred, self.outer_pipes = self.outer_blend.fit_predict(inner_preds, inner_pipes)
        else:
            val_pred = concatenate(inner_preds)
            self.outer_pipes = inner_pipes

        return val_pred

    def predict(
        self,
        data: Any,
        features_names: Optional[Sequence[str]] = None,
        return_all_predictions: Optional[bool] = None,
        **kwargs,
    ) -> LAMLDataset:
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

        Args:
            data: Dataset to perform inference.
            features_names: Optional features names,
              if cannot be inferred from `train_data`.
            return_all_predictions: bool - skip blending phase

        Returns:
            Dataset with predictions.

        """

        if return_all_predictions is None or self.return_all_predictions:
            return_all_predictions = self.return_all_predictions

        outer_preds = []

        for amls_pipe in self.outer_pipes:

            inner_preds = []
            # TODO: Maybe refactor?
            for automl in amls_pipe.ml_algos[0].models[0]:
                inner_pred = automl.predict(data, features_names, **kwargs)
                inner_preds.append(inner_pred)

            outer_pred = amls_pipe.blender.predict(inner_preds)
            outer_preds.append(outer_pred)

        # pred = self.outer_blend.predict(outer_preds)

        if not return_all_predictions:
            pred = self.outer_blend.predict(outer_preds)
        else:
            pred = concatenate(outer_preds)

        return pred
