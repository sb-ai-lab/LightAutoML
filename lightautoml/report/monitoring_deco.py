import os

from abc import ABC
from abc import abstractmethod
from collections import Counter
from typing import Any
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jinja2 import Environment
from jinja2 import FileSystemLoader

from .report_deco import ReportDeco


def abs_compare(value: np.ndarray, base: np.ndarray, threshold: np.ndarray, is_higher: bool) -> np.ndarray:
    return is_higher * (base - value) >= threshold


def rel_compare(value: np.ndarray, base: np.ndarray, threshold: np.ndarray, is_higher: bool) -> np.ndarray:
    sign = np.sign(base) * is_higher
    return is_higher * ((1 - sign * threshold) * base - value) >= 0


def check_shape(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        value = np.array(value)

    if len(value.shape) == 1:
        value = value.reshape(1, len(value))

    return value


class BaseDetector(ABC):
    """Base class for detection time-series anomalies."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @abstractmethod
    def predict(self, x: np.ndarray):
        pass

    @abstractmethod
    def fit_predict(self, x: np.ndarray):
        pass

    @abstractmethod
    def update(self, x: np.ndarray):
        pass


class ThresholdDetector(BaseDetector):
    _comps = {"relative": rel_compare, "absolute": abs_compare}
    _rules = {"any": np.any, "all": np.all, "raw": np.array}

    def __init__(
        self,
        threshold: Union[int, float, np.ndarray] = 0.2,
        higher_is_better: bool = True,
        comp: str = "absolute",
        rule: str = "all",
        store_values: bool = False,
        base: Optional = None,
        **kwargs: Any,
    ):
        """Simple threshold detector.

        Args:
            threshold: value to compare with, may me multi-dimension.
            higher_is_better: whether higher value is better.
            comp: comparison type of decision rule.
            rule: type of aggregation for multi-dimension series.
            store_values: save historical values of the series.
            base: specify base value without calling fit.

        .. note::
            There are different types of decision rule:
                - absolute: compare base and current value by their absolute values.
                The formula is: higher_is_better * (base - current) >= threshold.
                - relative: compare relative change of values.
                The formula is: higher_is_better * ((1 - sign(base) * higher_is_better * threshold) * base - current) >= 0.

        .. note::
            One can specify aggregation rule for multi-dimension arrays:
                - any: return True if one of the elements is True.
                - all: return True if all elements is True.
                - raw: return raw bool array.

        """
        super().__init__(**kwargs)
        if rule not in self._rules:
            raise ValueError("rule - {} - not in the list of available types {}".format(rule, list(self._rules.keys())))

        if comp not in self._comps:
            raise ValueError("type - {} - not in the list of available types {}".format(comp, list(self._comps.keys())))

        self.rule = self._rules[rule]
        self.comp = self._comps[comp]
        self.is_higher = 2 * higher_is_better - 1
        self.threshold = check_shape(threshold)
        self.store_values = store_values

        self.reset(reset_base=True)

        if base is not None:
            self.base = check_shape(base)

    def __repr__(self):
        return f"ThresholdDetector(threshold={self.threshold}, base={self.base}). Len history: {len(self.history)}"

    def fit(self, x: np.ndarray):
        """Store base metric value to compare with it.

        Args:
            x: vector with size (1, 1) or float for one 1d time-series and (1, m) for m-d time-series.
        """
        self.base = check_shape(x)
        return self

    def predict(self, x: np.ndarray, store: bool = True) -> Union[int, np.ndarray]:
        """Apply threshold based decision rule and save input if needed.

        Args:
            x: vector with size (1, 1) or float for one 1d time-series and (1, m) for m-d time-series.
            If input has size (n, m) then raw aggregation rule should be specified.
            store: force to not save history for current iteration if False.
        Returns:
            int or array with decision. One corresponds to detection.
        """
        x = check_shape(x)
        if self.store_values and store:
            self.x.append(x)
        decision = self.rule(self.comp(x, self.base, self.threshold, self.is_higher)).astype(int)

        return decision

    def fit_predict(self, x: np.ndarray):
        """Store base value and return dummy decision without saving history."""

        self.fit(x)
        return self.predict(x, store=False)

    def update(self, x: np.ndarray):
        """Make decision and save it in history."""

        pred = self.predict(x)
        self.history.append(pred)
        return self

    def undo(self):
        """Undo last update operation."""

        if len(self.history) >= 1:
            _ = self.history.pop()
            if self.store_values:
                _ = self.x.pop()

        return self

    def get_results(self) -> np.ndarray:
        """Return full decisions history."""

        return np.vstack(self.history) if len(self.history) > 0 else np.array([])

    def get_last_result(self) -> np.ndarray:
        """Return last decision."""

        return self.history[-1] if len(self.history) > 0 else None

    def get_values(self) -> np.ndarray:
        """Return full series history."""

        return np.vstack(self.x) if len(self.x) > 0 else np.array([])

    def reset(self, reset_base: bool = True):
        """Reset history.

        Args:
            reset_base: reset base value.
        """

        self.history = []
        self.x = []

        if reset_base:
            self.base = np.empty((1, 1))
        return self


class DataShiftDetector:
    def __init__(self, **kwargs):
        self.n_buckets = kwargs.get("n_buckets", 20)
        self.null_threshold = kwargs.get("null_threshold", 0.2)
        self.shift_threshold = kwargs.get("shift_threshold", 0.1)
        self.category_tol = kwargs.get("category_tol", 0.02)
        self.percent_eps = kwargs.get("percent_eps", 0.0001)
        self.breakpoints_tol_digits = 5
        self.n_epoch = 0
        self.data_info = {}
        self.alerts = {}
        self.results = {}

    def fit(self, train_data, roles):
        """
        + 1. Для каждой numeric фичи найти квантили и подсчитать гистограмму
        + 2. Для каждой categorical фичи получить числовой маппинг и подсчитать гистограмму
        + 3. Для каждой фичи подсчитать долю null
        """
        self.roles = roles
        for field_name in self.roles:
            if self.roles[field_name].name not in ["Numeric", "Category"]:
                continue
            field_series = train_data[field_name]
            self.data_info[field_name] = {}
            self.alerts[field_name] = {}
            self.results[field_name] = {}
            # share of null values
            null_values = np.sum(field_series.isnull().values) / len(field_series)
            self.data_info[field_name]["null_detector"] = ThresholdDetector(
                base=null_values, threshold=self.null_threshold, higher_is_better=False, store_values=True
            )
            # drop null
            field_series = field_series.dropna()
            if self.roles[field_name].name == "Numeric":
                self.data_info[field_name]["type"] = "Numeric"
                (
                    self.data_info[field_name]["breakpoints"],
                    self.data_info[field_name]["percents"],
                ) = self._breakpoints_percents(field_series.values)
                self.data_info[field_name]["shift_detector"] = ThresholdDetector(
                    base=0.0, threshold=self.shift_threshold, higher_is_better=False, store_values=True
                )

            if self.roles[field_name].name == "Category":
                self.data_info[field_name]["type"] = "Category"
                mapping, percents, categories_cnt = self._category_percents(field_series.values)
                self.data_info[field_name]["mapping"] = mapping
                self.data_info[field_name]["percents"] = percents
                self.data_info[field_name]["categories"] = categories_cnt

                self.data_info[field_name]["shift_detector"] = ThresholdDetector(
                    base=0.0, threshold=self.shift_threshold, higher_is_better=False, store_values=True
                )

    def _psi(self, expected_array, actual_array):
        def sub_psi(expected_perc, actual_perc):
            expected_perc = max(expected_perc, self.percent_eps)
            actual_perc = max(actual_perc, self.percent_eps)
            return (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)

        return np.array([sub_psi(e, a) for e, a in zip(expected_array, actual_array)]).sum()

    def _breakpoints_percents(self, feature_array):
        quantiles = np.arange(0, self.n_buckets + 1) / self.n_buckets * 100
        breakpoints = np.array([np.percentile(feature_array, q) for q in quantiles])
        breakpoints = np.unique(np.around(breakpoints, self.breakpoints_tol_digits))
        percents = np.histogram(feature_array, breakpoints)[0] / len(feature_array)
        return breakpoints, percents

    def _category_percents(self, feature_array):
        n_obj = len(feature_array)
        categories_cnt = Counter(feature_array)
        cnt = 0
        mapping = {}
        for value, num in categories_cnt.most_common():
            mapping[value] = cnt
            if num / n_obj >= self.category_tol:
                cnt += 1
        percents = {i: 0 for i in set(mapping.values())}
        for key in mapping:
            percents[mapping[key]] += categories_cnt[key] / n_obj
        return mapping, percents, categories_cnt

    def update(self, epoch_data):
        """
        + 1. Для каждой numeric фичи посчитать гистограмму по предрасчитанным значениям квантилей
        + 2. Для каждой numeric фичи подсчитать PSI с тренировочной гистограммой,
             если выходит за threshold, добавить в alerts
        + 3. Для каждой categorical фичи проверить не появилось ли новых значений, которых не было в обучении,
             если появились, добавить в alerts
        + 4. Для каждой categorical фичи подсчитать PSI с тренировочной гистограммой,
             если выходит за threshold, добавить в alerts
        + 5. Для каждой фичи подсчитать долю null. Если выходит за threshold, добавить в alerts.
        """
        for field_name in self.data_info:
            field_series = epoch_data[field_name]
            # share of null values
            null_values = np.sum(field_series.isnull().values) / len(field_series)
            self.data_info[field_name]["null_detector"].update(null_values)
            # null values: update alert
            if np.any(self.data_info[field_name]["null_detector"].get_results().ravel()):
                self.alerts[field_name]["null_values"] = (
                    self.data_info[field_name]["null_detector"].get_values().ravel()
                )
            # drop null
            field_series = field_series.dropna()

            # Numeric variable
            if self.roles[field_name].name == "Numeric":
                actual_percents = np.histogram(field_series, self.data_info[field_name]["breakpoints"])[0] / len(
                    field_series
                )
                psi = self._psi(self.data_info[field_name]["percents"], actual_percents)
                self.data_info[field_name]["shift_detector"].update(psi)

            # Category variable
            if self.roles[field_name].name == "Category":
                # expected percents
                expected_percents = self.data_info[field_name]["percents"]
                expected_percents = np.array([expected_percents[i] for i in sorted(expected_percents)])
                # actual_percents
                actual_percents = np.zeros_like(expected_percents)
                n_obj = len(field_series)
                for value, num in Counter(field_series).most_common():
                    if value in self.data_info[field_name]["mapping"]:
                        actual_percents[self.data_info[field_name]["mapping"][value]] += num / n_obj
                    else:
                        actual_percents[-1] += num / n_obj
                    if value not in self.data_info[field_name]["categories"]:
                        new_category_item = {
                            "n_epoch": self.n_epoch,
                            "value": value,
                            "occurance": round(num / n_obj, 6),
                        }
                        if "new_category" in self.alerts[field_name]:
                            self.alerts[field_name]["new_category"].append(new_category_item)
                        else:
                            self.alerts[field_name]["new_category"] = [new_category_item]
                psi = self._psi(expected_percents, actual_percents)
                self.data_info[field_name]["shift_detector"].update(psi)

            # psi: update alerts
            if np.any(self.data_info[field_name]["shift_detector"].get_results().ravel()):
                self.alerts[field_name]["psi_values"] = (
                    self.data_info[field_name]["shift_detector"].get_values().ravel()
                )
        self.n_epoch += 1

    def get_all_results(self):
        """
        - 1. PSI по всем фичам
        - 2. Частоты всех категорий всех категориальных переменных (включая новые появившиеся)
        - 3. Изменение доли null всех фичей
        """
        return self.data_info

    def get_alerts(self):
        """
        + 1. Alerts бывают следующих видов:
             + PSI по фиче (любого вида) выходит за threshold
             + Обнуления частот категорий
             + Доля null выходит за threshold
        - 2. Что показываем в каждом из случаев:
             - График PSI для каждой эпохи и горизонтальная линия threshold
             - График частотности новой категории. Если она встречается несколько эпох подряд,
               график будет нулевой до определённого момента, а затем будет рост значения
             - График частотности обнулившейся категории.
               На нём можно будет увидеть, насколько быстро "оборвались" значения.
             - График изменения доли null для фичи
        """
        return self.alerts


class ModelShiftDetector(DataShiftDetector):
    """
    Определяет сдвиг в предсказаниях модели.
    Для бинарной классификации и регрессии мониторится значение предсказания.
    Для многоклассовой классификации мониторится близость распределния классов к тому, которое было на обучении.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = kwargs.get("task", "binary")  # ['binary', 'reg', 'multiclass']
        self.psi_shift_threshold = kwargs.get("psi_shift_threshold", 0.1)
        self.model_info = {}

    def fit(self, oof_preds):
        # Получает вектор предсказаний на oof выборке и запоминает статистику
        if self.task in ["binary", "reg"]:
            self.model_info["oof_breakpoints"], self.model_info["oof_percents"] = self._breakpoints_percents(oof_preds)
        else:
            self.model_info["oof_mapping"], self.model_info["oof_percents"], _ = self._category_percents(oof_preds)
        self.model_info["psi_shift_detector"] = ThresholdDetector(
            base=self.psi_shift_threshold, threshold=self.shift_threshold, higher_is_better=False, store_values=True
        )
        self.model_info["mean_shift_detector"] = ThresholdDetector(
            base=np.mean(oof_preds), threshold=self.shift_threshold, higher_is_better=False, store_values=True
        )

    def update(self, epoch_preds):
        # Получает вектор предсказаний на эпохе, считает статистику и сравнивает с сохранённой
        if self.task in ["binary", "reg"]:
            actual_percents = np.histogram(epoch_preds, self.model_info["oof_breakpoints"])[0] / len(epoch_preds)
            psi = self._psi(self.model_info["oof_percents"], actual_percents)
            self.model_info["psi_shift_detector"].update(psi)

        else:
            expected_percents = self.model_info["oof_percents"]
            expected_percents = np.array([expected_percents[i] for i in sorted(expected_percents)])
            # actual_percents
            actual_percents = np.zeros_like(expected_percents)
            n_obj = len(epoch_preds)
            for value, num in Counter(epoch_preds).most_common():
                if value in self.model_info["oof_mapping"]:
                    actual_percents[self.model_info["oof_mapping"][value]] += num / n_obj
                else:
                    actual_percents[-1] += num / n_obj
                psi = self._psi(expected_percents, actual_percents)
                self.model_info["psi_shift_detector"].update(psi)
        self.model_info["mean_shift_detector"].update(np.mean(epoch_preds))

    def get_results(self):
        return self.model_info["shift_detector"].get_values().ravel()


def plot_psi_shift_detector(psi_values, thres, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(16, 10))

    plt.plot(np.arange(len(psi_values)), psi_values, color="blue", lw=2, label="PSI values test")
    plt.plot([0, len(psi_values) - 1], [thres, thres], color="red", lw=2, linestyle="--", label="threshold")
    plt.ylim([0, max(psi_values) + 0.05])
    plt.title("Feature shift detector")
    plt.xlabel("N_epoch")
    plt.ylabel("PSI")

    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_null_shift_detector(null_values, thres, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(16, 10))

    plt.plot(np.arange(len(null_values)), null_values, color="blue", lw=2, label="NULL percentage test")
    plt.plot(
        [0, len(null_values) - 1], [thres, thres], color="red", lw=2, linestyle="--", label="NULL percentage train"
    )
    plt.ylim([0, max(null_values) + 0.05])
    plt.title("NULL shift detector")
    plt.xlabel("N_epoch")
    plt.ylabel("NULL percentage")

    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_psi_prediction_shift(psi_values, thres, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(16, 10))

    plt.plot(np.arange(len(psi_values)), psi_values, color="blue", lw=2, label="PSI values test")
    plt.plot([0, len(psi_values) - 1], [thres, thres], color="red", lw=2, linestyle="--", label="threshold")
    plt.ylim([0, max(psi_values) + 0.05])
    plt.title("PSI prediction shift detector")
    plt.xlabel("N_epoch")
    plt.ylabel("PSI")

    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_mean_prediction_shift(mean_values, thres, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(16, 10))

    plt.plot(np.arange(len(mean_values)), mean_values, color="blue", lw=2, label="PSI values test")
    plt.plot([0, len(mean_values) - 1], [thres, thres], color="red", lw=2, linestyle="--", label="threshold")
    plt.ylim([0, max(mean_values) + 0.05])
    plt.title("Mean prediction shift detector")
    plt.xlabel("N_epoch")
    plt.ylabel("Mean prediction")

    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


class MonitoringDeco(ReportDeco):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._data_shift_path = "data_shift_section.html"
        self._data_shift_subsection_path = "data_shift_subsection.html"
        self.sections_order.append("data_shift")
        self._model_shift_path = "model_shift_section.html"
        self.sections_order.append("model_shift")
        self._n_epoch = 0

        self.null_threshold = kwargs.get("null_threshold", 0.1)
        self.feature_shift_threshold = kwargs.get("feature_shift_threshold", 0.1)
        self.model_shift_threshold = kwargs.get("model_shift_threshold", 0.02)

    def fit_predict(self, *args, **kwargs):
        oof_preds = super().fit_predict(*args, **kwargs)
        roles = self._model.reader._roles
        train_data = kwargs["train_data"] if "train_data" in kwargs else args[0]

        # TODO: initialize with parameters
        self.DSD = DataShiftDetector(null_threshold=self.null_threshold, shift_threshold=self.feature_shift_threshold)
        self.MSD = ModelShiftDetector(task=self.task, psi_shift_threshold=self.model_shift_threshold)

        self.DSD.fit(train_data, roles)
        self.MSD.fit(oof_preds.data.ravel())

        return oof_preds

    def update(self, epoch_data):
        # TODO: rewrite predict method
        # preds = super().predict(epoch_data)

        self._n_epoch += 1
        # get predictions
        preds = self._model.predict(epoch_data)

        self.DSD.update(epoch_data)
        self._generate_data_shift_section()

        self.MSD.update(preds.data.ravel())
        self._generate_model_shift_section()

        # self._generate_data_shift_section()
        # self._generate_model_shift_section()
        self.generate_report()

    def _generate_data_shift_info(self):
        """
        TODO: collect self._data_shift_info

        1. Цикл по содержимому alerts (по названиям переменных)
        2. Если нет данных, пропускаем
        3. Если есть ключ psi_values:
           - строим график в координатах X = range(len(psi_values)), Y = psi_values
           - пунктирная линия на высоте self.DSD.shift_threshold
        4. Если есть ключ null_values:
           - строим график в координатах X = range(len(null_values)), Y = null_values
           - пунктирная линия на высоте self.DSD.null_threshold

        """
        data_shift_info = []
        for feature_name in self.DSD.alerts:
            # title, psi_shift_graph, null_shift_graph
            feature_alert = self.DSD.alerts[feature_name]
            if any(name in feature_alert for name in ["psi_values", "null_values"]):
                data_shift_subsection = {}
                data_shift_subsection["title"] = feature_name
                if "psi_values" in feature_alert:
                    data_shift_subsection["psi_shift_graph"] = feature_name + "_psi_shift.png"
                    plot_psi_shift_detector(
                        psi_values=feature_alert["psi_values"],
                        thres=self.DSD.shift_threshold,
                        path=os.path.join(self.output_path, data_shift_subsection["psi_shift_graph"]),
                    )
                if "null_values" in feature_alert:
                    data_shift_subsection["null_shift_graph"] = feature_name + "_null_shift.png"
                    plot_null_shift_detector(
                        null_values=feature_alert["null_values"],
                        thres=self.DSD.data_info[feature_name]["null_detector"].base,
                        path=os.path.join(self.output_path, data_shift_subsection["null_shift_graph"]),
                    )
                data_shift_info.append(self._generate_data_shift_subsection(data_shift_subsection))
        return data_shift_info

    def _generate_model_shift_info(self):
        # TODO: Проверить корректность подсчёта psi для предсказаний. Почему такие большие значения?

        model_shift_info = {}
        model_shift_info["psi_prediction_shift"] = "psi_prediction_shift.png"
        model_shift_info["mean_prediction_shift"] = "mean_prediction_shift.png"

        plot_psi_prediction_shift(
            psi_values=self.MSD.model_info["psi_shift_detector"].get_values(),
            thres=self.MSD.model_info["psi_shift_detector"].base,
            path=os.path.join(self.output_path, model_shift_info["psi_prediction_shift"]),
        )
        plot_mean_prediction_shift(
            mean_values=self.MSD.model_info["mean_shift_detector"].get_values(),
            thres=self.MSD.model_info["mean_shift_detector"].base,
            path=os.path.join(self.output_path, model_shift_info["mean_prediction_shift"]),
        )
        return model_shift_info

    def _generate_data_shift_section(self):
        data_shift_info = self._generate_data_shift_info()
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        data_shift_section = env.get_template(self._data_shift_path).render(data_shift_info=data_shift_info)
        self._sections["data_shift"] = data_shift_section

    def _generate_data_shift_subsection(self, data_shift_subsection):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        return env.get_template(self._data_shift_subsection_path).render(data_shift_subsection)

    def _generate_model_shift_section(self):
        model_shift_info = self._generate_model_shift_info()
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        model_shift_section = env.get_template(self._model_shift_path).render(model_shift_info)
        self._sections["model_shift"] = model_shift_section
