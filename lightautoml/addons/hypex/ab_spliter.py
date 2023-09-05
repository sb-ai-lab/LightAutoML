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


#  Методика рачета описана в Campaign Perfomance Management – Методика A/B-тестирования v.2.0 (стр.29)
def calc_mde(test_group: pd.Series, control_group: pd.Series, reliability: float = 0.95, power: float = 0.8,) -> float:
    """Calculates MDE (Minimum Detectable Effect).

    MDE - minimal effect that can be statistically substantiated comparing the two groups
    Calculation method is described in "Campaign Performance Management" - A/B-testing methodology v.2.0 (p.29)

    Args:
        test_group: Target group
        control_group: Control group
        reliability: Level of statistical reliability, usually equals 0.95
        power: Statistical criterion power

    Returns:
         mde: Minimum detectable effect
    """
    m = stats.norm.ppf((1 + reliability) / 2) + stats.norm.ppf(power)

    n_test, n_control = len(test_group), len(control_group)
    proportion = n_test / (n_test + n_control)
    p = np.sqrt(1 / (proportion * (1 - proportion)))

    var_test, var_control = test_group.var(ddof=1), control_group.var(ddof=1)
    s = np.sqrt((var_test / n_test) + (var_control / n_control))

    mde = m * p * s

    return mde


def calc_sample_size(
    test_group: pd.Series, control_group: pd.Series, mde: Union[Iterable[float], float], significance: float = 0.05, power: float = 0.8,
) -> float:
    """Calculates minimal required number of test objects for test in the general group.

    Calculation method is described in "Campaign Performance Management" - A/B-testing methodology v.2.0 (p.14)

    Args:
        test_group: Target group
        control_group: Control group
        mde: Minimal detectable effect
        significance: Statistical significance level - type I error probability (usually 0.05)
        power: Statistical criterion power

    Returns:
        min_sample_size: Minimal size of the general group
    """
    if isinstance(mde, Iterable):
        z_alpha = norm.ppf((2 - significance) / 2)
        z_betta = norm.ppf(power)

        p1 = mde[0]
        p2 = mde[1]

        min_sample_size = (z_alpha + z_betta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
    else:
        test_std = test_group.std()
        control_std = control_group.std()

        test_proportion = len(test_group) / (len(test_group) + len(control_group))
        control_proportion = 1 - test_proportion

        d = ((norm.ppf(1 - significance / 2) + norm.ppf(power)) / mde) ** 2

        s = test_std ** 2 / test_proportion + control_std ** 2 / control_proportion

        min_sample_size = d * s

    return min_sample_size


# --------------------- Classes ---------------------
class ABSplitter:
    """Abstract class - divider on A and B groups."""

    def __init__(self, mode="simple", by_group=None, quant_field=None):
        self.mode = mode
        self.by_group = by_group
        self.quant_field = quant_field

    @staticmethod
    def merge_groups(
        test_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
        control_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
    ):
        """Merges test and control groups in one DataFrame.

        Args:
            test_group: Data of target group
            control_group: Data of control group

        Returns:
            merged_data: Concatted DataFrame
        """
        if not (isinstance(test_group, pd.DataFrame) and isinstance(test_group, pd.DataFrame)):
            test_group = pd.concat(test_group, ignore_index=True)
            control_group = pd.concat(control_group, ignore_index=True)

        test_group.loc[:, "group"] = "test"
        control_group.loc[:, "group"] = "control"

        merged_data = pd.concat([test_group, control_group], ignore_index=True)

        return merged_data

    def __simple_mode(self, data: pd.DataFrame, random_state: int = None):
        result = {"test_indexes": [], "control_indexes": []}

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

    def split_ab(self, data: pd.DataFrame, random_state: int = None) -> Dict:
        """Divides sample on two groups.

        Args:
            data: input data
            random_state: one integer to fix split

        Returns:
            result: dict of indexes with division on test and control group
        """
        result = {"test_indexes": [], "control_indexes": []}

        if self.by_group:
            groups = data.groupby()
            for _, gd in groups:
                if self.mode not in ("balanced", "simple"):
                    warnings.warn(
                        f"The mode '{self.mode}' is not supported for group division. "
                        f"Implemented mode 'stratification'."
                    )
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
                warnings.warn(
                    f"The mode '{self.mode}' is not supported for regular division. "
                    f"Implemented mode 'simple'."
                )

            t_result = self.__simple_mode(data, random_state)
            result["test_indexes"] = t_result["test_indexes"]
            result["control_indexes"] = t_result["control_indexes"]

        result["test_indexes"] = list(set(result["test_indexes"]))
        result["control_indexes"] = list(set(result["test_indexes"]))

        return result

    def search_dist_uniform_sampling(
        self,
        data: pd.DataFrame,
        target_fields: Union[List[str], str],
        n: int = None,
        random_states: Iterable[int] = None,
        alpha: float = 0.05,
        file_name: Union[Path, str] = None,
        write_mode: str = "full",
        write_step: int = 10,
        pbar: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Chooses random_state for finding homogeneous distribution.

        Args:
            data:
                Input data
            target_fields:
                Field with target value
            n:
                Number of searching iterations
            random_states:
                Random states from searching (if given, n is ignoring)
            alpha:
                Threshold to check statistical hypothesis; usually 0.05
            file_name:
                Name of file to save results (if None - no results will be saved, func returns result)
            write_mode:
                Mode to write:
                    'full' - save all experiments
                    'all' - save experiments that passed all statistical tests
                    'any' - save experiments that passed any statistical test
            write_step:
                Step to write experiments to file
            pbar:
                Flag to show progress bar

        Returns:
             results:
                If no saving (no file_name, no write mode and no write_step) returns dataframe
                else None and saves file to csv
        """
        if random_states is None and n:
            random_states = range(n)

        results = []

        if write_mode not in ("full", "all", "any"):
            warnings.warn(
                f"Write mode '{write_mode}' is not supported. Mode 'full' will be used"
            )
            write_mode = "full"

        if isinstance(target_fields, str):
            target_fields = [target_fields]

        for i, random_state in tqdm(enumerate(random_states), total=len(random_states), display=pbar):
            split = self.split_ab(data, random_state)
            t_result = {"random_state": random_state}
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
                    pd.DataFrame(results).to_csv(file_name, index=False, header=False, mode="a")
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
    """Abstract class of A/B experiment."""

    def __init__(self, label: str):
        self.label = label

    @abstractmethod
    def calc_effect(self, test_data: pd.DataFrame, control_data: pd.DataFrame, target_field: str) -> float:
        pass


class ABTester:
    DEFAULT_FORMAT_MAPPING = {
        "rs": "random state",
        "mde": "MDE",
        "sample_size": "Size of test sample",
        "a_len": "Size of target group",
        "b_len": "Size of control group",
        "a_mean": "Mean of target group",
        "b_mean": "Mean of control group",
    }

    def __init__(
        self, splitter: ABSplitter, target_field: str, reliability=0.95, power=0.8, mde=None,
    ):
        """
        Args:
            splitter:
                Class of divider on A and B groups
            target_field:
                Field with target values
            reliability:
                Level of statistical reliability, usually equals 0.95
            power:
                Statistical criterion power, usually equals 0.8
            mde:
                Calculated mde (minimal detected effect),
                if none - calculates inside
        """
        self.splitter = splitter
        self.target_field = target_field
        self.reliability = reliability
        self.power = power
        self.mde = mde

    def sampling_test(
        self, data: pd.DataFrame, experiments: Union[ABExperiment, Iterable[ABExperiment]], random_state: int = None,
    ) -> Dict:
        """Test on specific sample.

        Args:
            data: Input data
            experiments: Experiment or set of experiments applied on sample
            random_state: Seed of random

        Returns:
            result: Test results
        """

        split = self.splitter.split_ab(data, random_state)
        if isinstance(experiments, ABExperiment):
            experiments = [experiments]

        mde = self.mde or calc_mde(
            data.loc[split["test"], self.target_field],
            data.loc[split["control"], self.target_field],
            reliability=self.reliability,
            power=self.power,
        )
        sample_size = calc_sample_size(
            data.loc[split["test"], self.target_field],
            data.loc[split["control"], self.target_field],
            mde,
            significance=(1 - self.reliability),
            power=self.power,
        )

        result = {
            "rs": random_state,
            "mde": mde,
            "sample_size": sample_size,
            "a_len": len(split["test"]),
            "b_len": len(split["control"]),
            "a_mean": data.loc[split["test"], self.target_field].mean(),
            "b_mean": data.loc[split["control"], self.target_field].mean(),
        }

        for e in experiments:
            result[f"effect {e.label}"] = e.calc_effect(
                data.loc[split["test"]], data.loc[split["control"]], self.target_field
            )

        return result

    def multisampling_test(
        self,
        data: pd.DataFrame,
        experiments: Union[ABExperiment, Iterable[ABExperiment]],
        random_states: Iterable[int],
        pbar: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Implements multiple experiments on random states.

        Args:
            data:
                Input data
            experiments:
                Set of experiments applied on sample
            random_states:
                Seeds of random
            pbar:
                Flag to show progress bar

        Returns:
            results:
                Experiment test results
            statistics:
                Description statistics
        """

        results = pd.DataFrame([self.sampling_test(data, experiments, rs) for rs in tqdm(random_states, display=pbar)])

        statistics = results.describe()
        statistics.loc["cv %"] = (statistics.loc["std"] / statistics.loc["mean"] * 100).round(2)
        return results, statistics

    def format_stat(
        self, stat: pd.DataFrame, experiments: Union[ABExperiment, Iterable[ABExperiment]], rename_map: Dict = None,
    ):
        """Corrects format of output statistics.

        Args:
            stat: Experiment statistics
            experiments: Set of experiments applied on sample
            rename_map: Mapping of renaming fields

        Returns:
            result: Formatted values
        """
        rename_map = rename_map or self.DEFAULT_FORMAT_MAPPING

        rename_map.update({f"effect {e.label}": f"Effect {e.label}" for e in experiments})

        result = stat.rename(columns=rename_map)
        result = result.applymap(lambda x: f"{x:,.2f}")
        return result
