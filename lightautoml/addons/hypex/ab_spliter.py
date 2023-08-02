# v: 0.2.1a
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.utils import shuffle
from typing import Iterable, Union, Optional, List, Dict

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, ttest_ind, ks_2samp


def calc_mde(
        test_group: pd.Series,
        control_group: pd.Series,
        reliability: float = 0.95,
        power: float = 0.8,
) -> float:
    """
    Minimum detectable effect
    Минимальный эффект, который можно статистически обосновать при сравнении двух групп
    Методика рачета описана в Campaign Perfomance Management – Методика A/B-тестирования v.2.0 (стр.29)

    :param test_group: целевая группа
    :param control_group: контрольная группа
    :param reliability: уровень статистической достоверности. Обычно равен 0.95
    :param power: мощность статистического критерия
    :return: минимально детерминируемый эффект
    """
    m = stats.norm.ppf((1 + reliability) / 2) + stats.norm.ppf(power)

    n_test, n_control = len(test_group), len(control_group)
    proportion = n_test / (n_test + n_control)
    p = np.sqrt(1 / (proportion * (1 - proportion)))

    var_test, var_control = test_group.var(ddof=1), control_group.var(ddof=1)
    s = np.sqrt((var_test / n_test) + (var_control / n_control))

    return m * p * s


def calc_sample_size(
        test_group: pd.Series,
        control_group: pd.Series,
        mde,
        significance: float = 0.05,
        power: float = 0.8,
) -> float:
    """
    Минимально требуемуе количество объектов тестирования в общей группе
    Методика расчета описана в Campaign Perfomance Management – Методика A/B-тестирования v.2.0 (стр.14)

    :param test_group: целевая группа
    :param control_group: контрольная группа
    :param mde: минимальное детерменируемое значение
    :param significance: уровень статистической значимости - вероятность ошибки первого рода (обычно 0.05)
    :param power: мощность статистического критерия
    :return: минимальный размер общей группы
    """
    test_std = test_group.std()
    control_std = control_group.std()

    test_proportion = len(test_group) / (len(test_group) + len(control_group))
    control_proportion = 1 - test_proportion

    d = ((norm.ppf(1 - significance / 2) + norm.ppf(power)) / mde) ** 2

    s = test_std ** 2 / test_proportion + control_std ** 2 / control_proportion

    return d * s


# --------------------- Classes ---------------------
class ABSplitter:
    """
    Класс разделителя на A|B группы
    """

    def __init__(self, mode="simple", by_group=None, quant_field=None):
        self.mode = mode
        self.by_group = by_group
        self.quant_field = quant_field

    @staticmethod
    def merge_groups(
            test_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
            control_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
    ):
        """
        Объединяет test и control в один df

        :return: объединенный датафрейм
        """
        if not (
                isinstance(test_group, pd.DataFrame)
                and isinstance(test_group, pd.DataFrame)
        ):
            test_group = pd.concat(test_group, ignore_index=True)
            control_group = pd.concat(control_group, ignore_index=True)

        test_group.loc[:, "group"] = "test"
        control_group.loc[:, "group"] = "control"
        return pd.concat([test_group, control_group], ignore_index=True)

    def __simple_mode(self, data, random_state):
        result = {
            "test_indexes": [],
            "control_indexes": []
        }

        if self.quant_field:
            random_ids = shuffle(data[self.quant_field].unique(), random_state=random_state)
            edge = len(random_ids) // 2
            result["test_indexes"] = list(data[data[self.quant_field].isin(random_ids[:edge])].index)
            result["control_indexes"] = list(data[data[self.quant_field].isin(random_ids[edge:])].index)

        else:
            addition_indexes = list(shuffle(data.index, random_state=random_state))
            edge = len(addition_indexes) // 2
            result["test_indexes"] = addition_indexes[:edge]
            result["control_indexes"] = addition_indexes[edge:]

        return result

    def split_ab(self, data, random_state: int = None) -> Dict:
        result = {
            "test_indexes": [],
            "control_indexes": []
        }

        if self.by_group:
            groups = data.groupby()
            for _, gd in groups:
                if self.mode not in ("balanced", "simple"):
                    warnings.warn(f"Не предусмотрено режима '{self.mode}' для группового разделения. "
                                  f"Был использован режим 'stratification'.")
                    self.mode = "simple"

                if self.mode == "simple":
                    t_result = self.__simple_mode(gd, random_state)
                    result["test_indexes"] += t_result["test_indexes"]
                    result["control_indexes"] += t_result["control_indexes"]

                elif self.mode == "balanced":
                    if self.quant_field:
                        random_ids = shuffle(gd[self.quant_field].unique(), random_state=random_state)
                        addition_indexes = list(gd[gd[self.quant_field].isin(random_ids)].index)
                    else:
                        addition_indexes = list(shuffle(gd.index, random_state=random_state))

                    if len(result["control_indexes"]) > len(result["test_indexes"]):
                        result["test_indexes"] += addition_indexes
                    else:
                        result["control_indexes"] += addition_indexes

        else:
            if self.mode != "simple":
                warnings.warn(f"Не предусмотрено режима '{self.mode}' для обычного разделения. "
                              f"Был использован режим 'simple'.")

            t_result = self.__simple_mode(data, random_state)
            result["test_indexes"] = t_result["test_indexes"]
            result["control_indexes"] = t_result["control_indexes"]

        result["test_indexes"] = list(set(result["test_indexes"]))
        result["control_indexes"] = list(set(result["test_indexes"]))
        return result

    def search_dist_uniform_sampling(
            self,
            data,
            target_fields: Union[List[str], str],
            n: int = None,
            random_states: Iterable[int] = None,
            alpha: float = 0.05,
            file_name: Union[Path, str] = None,
            write_mode: str = "full",
            write_step: int = 10,
            pbar: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Подбирает random_state для поиска однородного распределения

        :param target_field: поле с целевым значением
        :param n: количество итераций поиска
        :param random_states: случайные состояния по которым проводится поиск (альтернатива n, если введено, то n игнорируется)
        :param alpha: порог для проверки статистических гипотез
        :param file_name: имя файла, в котором будет сохраняться результат (если не заполнено, функция вернет результат, а не будет сохранять его в файл)
        :param write_mode: режим записи. Поддерживаются следующие:
            'full' - записывает все эксперименты
            'all' - записывает те эксперименты, которые прошли все статистические тесты
            'any' - записывает те эксперименты, которые прошли любой из статистических тестов
        :param write_step: шаг записи экспериментов в файл (если не указано, пошаговая запись не используется)
        :param pbar: отображать ли progress bar
        :return: DataFrame, если не использовалась пошаговая запись, иначе None
        """
        if random_states is None and n:
            random_states = range(n)

        results = []

        if write_mode not in ("full", "all", "any"):
            warnings.warn(
                f"Режим записи '{write_mode}' не поддерживается. Будет использован режим 'full'"
            )
            write_mode = "full"

        if isinstance(target_fields, str):
            target_fields = [target_fields]

        for i, random_state in tqdm(enumerate(random_states), total=len(random_states), display=pbar):
            split = self.split_ab(data, random_state)
            t_result = {
                "random_state": random_state
            }
            a = data.loc[split["test_indexes"]]
            b = data.loc[split["control_indexes"]]
            scores = []
            passed = []

            for tf in target_fields:
                ta = a[tf]
                tb = b[tf]

                t_result[f"{tf} a mean"] = ta.mean()
                t_result[f"{tf} b mean"] = tb.mean()
                t_result[f"{tf} ab mran delta %"] = (1 - t_result[f"{tf} a mean"] / t_result[f"{tf} b mean"]) * 100
                t_result[f"{tf} t_test p_value"] = ttest_ind(ta, tb).pvalue
                t_result[f"{tf} ks_test p_value"] = ks_2samp(ta, tb).pvalue
                t_result[f"{tf} t_test passed"] = t_result[f"{tf} t_test p_value"] > alpha
                t_result[f"{tf} ks_test passed"] = t_result[f"{tf} ks_test p_value"] > alpha
                scores.append((t_result[f"{tf} t_test p_value"] + t_result[f"{tf} ks_test p_value"]) / 2)
                passed += [t_result[f"{tf} t_test passed"], t_result[f"{tf} ks_test passed"]]

            t_result["score"] = np.mean(scores)

            if write_mode == "all" and all(passed):
                results.append(t_result)
            if write_mode == "any" and any(passed):
                results.append(t_result)
            if write_mode == "full":
                results.append(t_result)

            if file_name and write_step:
                if i == write_step:
                    pd.DataFrame(results).to_csv(file_name, index=False)
                elif i % write_step == 0:
                    pd.DataFrame(results).to_csv(
                        file_name, index=False, header=False, mode="a"
                    )
                    results = []
        if file_name and write_step:
            pd.DataFrame(results).to_csv(file_name, index=False, header=False, mode="a")
        elif file_name:
            results = pd.DataFrame(results)
            results.to_csv(file_name, index=False)
            return results
        else:
            return pd.DataFrame(results)


class ABExperiment(ABC):
    """
    Абстрактный класс A|B эксперимента
    """

    def __init__(self, label: str):
        self.label = label

    @abstractmethod
    def calc_effect(
            self, test_data: pd.DataFrame, control_data: pd.DataFrame, target_field: str
    ) -> float:
        pass


class ABTester:
    DEFAULT_FORMAT_MAPPING = {
        "rs": "random state",
        "mde": "MDE",
        "sample_size": "Размер выборки для тестирования",
        "a_len": "Размер целевой группы",
        "b_len": "Размер контрольной группы",
        "a_mean": "Среднее целевой группы",
        "b_mean": "Среднее контрольной группы",
    }

    def __init__(
            self,
            splitter: ABSplitter,
            target_field: str,
            reliability=0.95,
            power=0.8,
            mde=None,
    ):
        """
        :param splitter: класс разделителя на A|B
        :param target_field: поле с целевыми значениями
        :param reliability: уровень статистической достоверности. Обычно равен 0.95
        :param power: мощность статистического критерия. Обычно равен 0.8
        :param mde: предпосчитанный mde, если None, то считается
        """
        self.splitter = splitter
        self.target_field = target_field
        self.reliability = reliability
        self.power = power
        self.mde = mde

    def sampling_test(
            self,
            data,
            experiments: Union[ABExperiment, Iterable[ABExperiment]],
            random_state: int = None,
    ) -> Dict:
        """
        Тест на определенном разбиении
        :param experiments: эксперимент или набор экспериментов, проводимых на разбиении
        :random_state: seed рандома

        :return: dict с результатами теста
        """

        split = self.splitter.split_ab(data, random_state)
        if isinstance(experiments, ABExperiment):
            experiments = [experiments]

        mde = self.mde or calc_mde(
            data.loc[split['test'], self.target_field],
            data.loc[split['control'], self.target_field],
            reliability=self.reliability,
            power=self.power,
        )
        sample_size = calc_sample_size(
            data.loc[split['test'], self.target_field],
            data.loc[split['control'], self.target_field],
            mde,
            significance=(1 - self.reliability),
            power=self.power,
        )

        result = {
            "rs": random_state,
            "mde": mde,
            "sample_size": sample_size,
            "a_len": len(split['test']),
            "b_len": len(split['control']),
            "a_mean": data.loc[split['test'], self.target_field].mean(),
            "b_mean": data.loc[split['control'], self.target_field].mean(),
        }

        for e in experiments:
            result[f"effect {e.label}"] = e.calc_effect(
                data.loc[split['test']], data.loc[split['control']], self.target_field
            )

        return result

    def multisampling_test(
            self,
            data,
            experiments: Union[ABExperiment, Iterable[ABExperiment]],
            random_states: Iterable[int],
            pbar: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Проводит множественные эксперименты по случайным состояниям
        :param experiments: набор экспериментов, проводимых на разбиении
        :param random_states: случайные состояния
        :param pbar: активация прогресс бара
        :return: статистики экспериментов
        """

        results = pd.DataFrame(
            [
                self.sampling_test(data, experiments, rs)
                for rs in tqdm(random_states, display=pbar)
            ]
        )

        stats = results.describe()
        stats.loc["cv %"] = (stats.loc["std"] / stats.loc["mean"] * 100).round(2)
        return results, stats

    def format_stat(
            self,
            stat: pd.DataFrame,
            experiments: Union[ABExperiment, Iterable[ABExperiment]],
            rename_map: Dict = None,
    ):
        """
        Редактирует формат вывода статистик

        :param stat: статистики экспериментов
        :param experiments: набор экспериментов, проводимых на разбиении
        :param rename_map: маппинг переименования полей

        :return: форматирует датафрейм со статистиками
        """
        rename_map = rename_map or self.DEFAULT_FORMAT_MAPPING

        rename_map.update(
            {f"effect {e.label}": f"Эффект {e.label}" for e in experiments}
        )

        result = stat.rename(columns=rename_map)
        result = result.applymap(lambda x: f"{x:,.2f}")
        return result
