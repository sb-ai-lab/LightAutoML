import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.utils import shuffle
from typing import Iterable, Union, Optional, List, Dict, Any

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, ttest_ind, ks_2samp

RANDOM_STATE = 52


def merge_groups(
    test_group: Union[Iterable[pd.DataFrame], pd.DataFrame], control_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
):
    """Merges test and control groups in one DataFrame and creates column "group".

    Column "group" contains of "test" and "control" values.

    Args:
        test_group: Data of target group
        control_group: Data of control group

    Returns:
        merged_data: Concatted DataFrame
    """
    # if not isinstance(test_group, pd.DataFrame):
    #     test_group = pd.concat(test_group, ignore_index=True)
    # if not isinstance(control_group, pd.DataFrame):
    #     control_group = pd.concat(control_group, ignore_index=True)

    test_group.loc[:, "group"] = "test"
    control_group.loc[:, "group"] = "control"

    merged_data = pd.concat([test_group, control_group], ignore_index=True)

    return merged_data


class AATest:
    def __init__(
        self,
        data: pd.DataFrame,
        target_fields: Union[Iterable[str], str],
        info_cols: Union[Iterable[str], str] = None,
        group_cols: Union[str, Iterable[str]] = None,
        quant_field: str = None,
        mode: str = "simple"
    ):
        """

        Args:
            data:
                Input data
            target_fields:
                Field with target value
            mode:
                Regime to divide sample on A and B samples:
                    'simple' - divides by groups, placing equal amount
                    of records (clients | groups of clients) in samples
                    'balanced' - divides by size of samples, placing full groups depending on the size of group
                    to balance size of A and B. Can not be applied without groups
            group_cols:
                Name of field(s) for division by groups
            quant_field:
                Name of field by which division should take in account common features besides groups
        """
        self.data = data
        self.init_data = data
        self.target_fields = [target_fields] if isinstance(target_fields, str) else target_fields
        self.info_cols = [info_cols] if isinstance(info_cols, str) else info_cols
        self.group_cols = [group_cols] if isinstance(group_cols, str) else group_cols
        self.quant_field = quant_field
        self.mode = mode
        self._preprocessing_data()

    def _preprocessing_data(self):
        """Converts categorical variables to dummy variables.

        Returns:
            Data with categorical variables converted to dummy variables.
        """
        data = self.data

        # categorical to dummies
        init_cols = data.columns

        dont_binarize_cols = (  # collects names of columns that shouldn't be binarized
            self.group_cols+[self.quant_field]
            if (self.group_cols is not None) and (self.quant_field is not None)
            else self.group_cols
            if self.group_cols is not None
            else [self.quant_field]
            if self.quant_field is not None
            else None
        )
        # if self.group_cols is not None:
        if dont_binarize_cols is not None:
            data = pd.get_dummies(data.drop(columns=dont_binarize_cols), dummy_na=True)
            data = data.merge(self.data[dont_binarize_cols], left_index=True, right_index=True)
        else:
            data = pd.get_dummies(data, dummy_na=True)

        # fix if dummy_na is const=0
        dummies_cols = set(data.columns) - set(init_cols)
        const_columns = [col for col in dummies_cols if data[col].nunique() <= 1]  # choose constant_columns
        cols_to_drop = const_columns + (self.info_cols if self.info_cols is not None else [])
        self.data = data.drop(columns=cols_to_drop)

    def __simple_mode(self, data: pd.DataFrame, random_state: int = RANDOM_STATE):
        """Separates data on A and B samples within simple mode.

        Separation performed to divide groups of equal sizes - equal amount of records
        or equal amount of groups in each sample.

        Args:
            data: Input data
            random_state: Seed of random

        Returns:
            result: Test and control samples of indexes dictionary
        """
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

    def split_ab(self, random_state: int = RANDOM_STATE) -> Dict:
        """Divides sample on two groups.

        Args:
            random_state: one integer to fix split

        Returns:
            result: dict of indexes with division on test and control group
        """
        data = self.data
        result = {"test_indexes": [], "control_indexes": []}

        if self.group_cols:
            groups = data.groupby(self.group_cols)
            for _, gd in groups:
                if self.mode not in ("balanced", "simple"):
                    warnings.warn(
                        f"The mode '{self.mode}' is not supported for group division. Implemented mode 'simple'."
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
                    f"The mode '{self.mode}' is not supported for regular division. " f"Implemented mode 'simple'."
                )

            t_result = self.__simple_mode(data, random_state)
            result["test_indexes"] = t_result["test_indexes"]
            result["control_indexes"] = t_result["control_indexes"]

        result["test_indexes"] = list(set(result["test_indexes"]))
        result["control_indexes"] = list(set(result["test_indexes"]))

        return result

    def _postprep_data(self, spit_indexes: Dict = None):
        """prep data to show user (add info_cols and decode binary variables)

        Args:
            spit_indexes: dict of indexes with separation on test and control group

        Returns:
            data: separated init data with column "group"
        """
        # prep data to show user (add info_cols and decode binary variables)
        test = self.init_data.loc[spit_indexes["test_indexes"]]
        control = self.init_data.loc[spit_indexes["control_indexes"]]
        data = merge_groups(test, control)

        return data

    def sampling_metrics(self, alpha: float = 0.05, random_state: int = RANDOM_STATE):
        """

        Args:
            alpha: Threshold to check statistical hypothesis; usually 0.05
            random_state: Random seeds for searching

        Returns:
            result: tuple of
                1) metrics dataframe (stat tests) and
                2) dict of random state with test_control dataframe
        """
        data_from_sampling_dict = {}
        scores = []
        t_result = {"random_state": random_state}

        split = self.split_ab(random_state)
        a = self.data.loc[split["test_indexes"]]
        b = self.data.loc[split["control_indexes"]]

        # prep data to show user (merge indexes and init data)
        data_from_sampling_dict[random_state] = self._postprep_data(split)

        for tf in self.target_fields:
            ta = a[tf]
            tb = b[tf]

            t_result[f"{tf} a mean"] = ta.mean()
            t_result[f"{tf} b mean"] = tb.mean()
            t_result[f"{tf} ab delta %"] = (1 - t_result[f"{tf} a mean"] / t_result[f"{tf} b mean"]) * 100
            t_result[f"{tf} t_test p_value"] = ttest_ind(ta, tb).pvalue
            t_result[f"{tf} ks_test p_value"] = ks_2samp(ta, tb).pvalue
            t_result[f"{tf} t_test passed"] = t_result[f"{tf} t_test p_value"] > alpha
            t_result[f"{tf} ks_test passed"] = t_result[f"{tf} ks_test p_value"] > alpha
            scores.append((t_result[f"{tf} t_test p_value"] + t_result[f"{tf} ks_test p_value"]) / 2)

        t_result["mean_tests_score"] = np.mean(scores)
        result = {"metrics": t_result, "data_from_experiment": data_from_sampling_dict}

        return result

    def search_dist_uniform_sampling(
        self,
        alpha: float = 0.05,
        iterations: int = 10,
        file_name: Union[Path, str] = None,
        write_mode: str = "full",
        write_step: int = None,
        pbar: bool = True,
    ) -> Optional[tuple[pd.DataFrame, dict[Any, dict]]]:
        """Chooses random_state for finding homogeneous distribution.

        Args:
            iterations:
                Number of iterations to search uniform sampling to searching
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
        random_states = range(iterations)
        results = []
        data_from_sampling = {}

        if write_mode not in ("full", "all", "any"):
            warnings.warn(f"Write mode '{write_mode}' is not supported. Mode 'full' will be used")
            write_mode = "full"

        for i, rs in tqdm(enumerate(random_states), total=len(random_states)):#, display=pbar):
            res = self.sampling_metrics(alpha=alpha, random_state=rs)
            data_from_sampling.update(res["data_from_experiment"])

            # write to file
            passed = []
            for tf in self.target_fields:
                passed += [res["metrics"][f"{tf} t_test passed"], res["metrics"][f"{tf} ks_test passed"]]

            if write_mode == "all" and all(passed):
                results.append(res["metrics"])
            if write_mode == "any" and any(passed):
                results.append(res["metrics"])
            if write_mode == "full":
                results.append(res["metrics"])

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
            return results, data_from_sampling
        else:
            return pd.DataFrame(results), data_from_sampling


# class ABTest:
#     """Calculates metrics - MDE, ATE and p_value."""
#
#     DEFAULT_FORMAT_MAPPING = {
#         "rs": "random state",
#         "mde": "MDE",
#         "sample_size": "Size of test sample",
#         "a_len": "Size of target group",
#         "b_len": "Size of control group",
#         "a_mean": "Mean of target group",
#         "b_mean": "Mean of control group",
#     }
#
#     def __init__(
#         self, target_field: str, reliability: float = 0.95, power: float = 0.8, mde: float = None,
#     ):
#         """
#         Args:
#             splitter:
#                 Class of divider on A and B groups
#             target_field:
#                 Field with target values
#             reliability:
#                 Level of statistical reliability, usually equals 0.95
#             power:
#                 Statistical criterion power, usually equals 0.8
#             mde:
#                 Calculated mde (minimal detected effect),
#                 if none - calculates inside
#         """
#         self.test = test_data
#         self.control = control_data
#         self.target_field = target_field
#         self.reliability = reliability
#         self.power = power
#         self.mde = mde
#
#     def sampling_test(
#         self, data: pd.DataFrame, experiments = None, random_state: int = None,
#     ) -> Dict:
#         """Test on specific sample.
#
#         Args:
#             data: Input data
#             experiments: Experiment or set of experiments applied on sample
#             random_state: Seed of random
#
#         Returns:
#             result: Test results
#         """
#
#         # split = self.splitter.split_ab(data, random_state)
#         # if isinstance(experiments, ABExperiment):
#         #     experiments = [experiments]
#
#         mde = self.mde or calc_mde(
#             data.loc[self.test, self.target_field],
#             data.loc[self.control, self.target_field],
#             reliability=self.reliability,
#             power=self.power,
#         )
#         sample_size = calc_sample_size(
#             data.loc[self.test, self.target_field],
#             data.loc[self.control, self.target_field],
#             mde,
#             significance=(1 - self.reliability),
#             power=self.power,
#         )
#
#         result = {
#             "rs": random_state,
#             "mde": mde,
#             "sample_size": sample_size,
#             "a_len": len(self.test),
#             "b_len": len(self.control),
#             "a_mean": data.loc[self.test, self.target_field].mean(),
#             "b_mean": data.loc[self.control, self.target_field].mean(),
#         }
#         #  включить класс из ноута
#         for e in experiments: #: как считается эффект написано в эксперименте, перенести в calc_effect
#             """
#             сделать разницу средних в наследнике класса (новый класс создать)
#             на альфе в к7м ABTesting, IncraceExperiment
#             передается эксперимент, (надо встроить эксперимент сюда)
#             целевая картинка - передать данные и получить результат
#             сейчас надо вшить эксперимент из ноутбука сюда
#             """
#             result[f"effect {e.label}"] = e.calc_effect(
#                 data.loc[split["test"]], data.loc[split["control"]], self.target_field
#             )
#
#         return result
#
#     def format_stat(
#         self, stat: pd.DataFrame, experiments = None, rename_map: Dict = None,
#     ):
#         """Corrects format of output statistics.
#
#         Args:
#             stat: Experiment statistics
#             experiments: Set of experiments applied on sample
#             rename_map: Mapping of renaming fields
#
#         Returns:
#             result: Formatted values
#         """
#         rename_map = rename_map or self.DEFAULT_FORMAT_MAPPING
#
#         rename_map.update({f"effect {e.label}": f"Effect {e.label}" for e in experiments})
#
#         result = stat.rename(columns=rename_map)
#         result = result.applymap(lambda x: f"{x:,.2f}")
#         return result
