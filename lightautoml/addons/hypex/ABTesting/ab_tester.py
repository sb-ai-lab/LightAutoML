import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.utils import shuffle
from typing import Iterable, Union, Optional, List, Dict, Any

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, ttest_ind, ks_2samp, mannwhitneyu

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


class ABTest:
    def __init__(
        self,
        calc_difference_method: str = "all",
        calc_p_value_method: str = "all",
    ):
        """
        Initializes the ABTest class.
        Parameters:
            calc_difference_method (str, optional): The method used to calculate the difference. Defaults to 'all'.
            calc_p_value_method (str, optional): The method used to calculate the p-value. Defaults to 'all'.
        """
        self.calc_difference_method = calc_difference_method
        self.calc_p_value_method = calc_p_value_method

    def split_ab(self, data: pd.DataFrame, group_field: str) -> Dict[str, pd.DataFrame]:
        """
        Splits a pandas DataFrame into two separate dataframes based on a specified group field.

        Parameters:
            data (pd.DataFrame): The input dataframe to be split.
            group_field (str): The column name representing the group field.

        Returns:
            dict: A dictionary containing two dataframes, 'test' and 'control', where 'test' contains rows where the group field is 'test', and 'control' contains rows where the group field is 'control'.
        """
        return {
            "test": data[data[group_field] == "test"],
            "control": data[data[group_field] == "control"],
        }

    def calc_difference(
        self, splitted_data: Dict[str, pd.DataFrame], target_field: str
    ) -> Dict[str, float]:
        """
        Calculates the difference between the target field values of the 'test' and 'control' dataframes.

        Parameters:
            splitted_data (Dict[str, pd.DataFrame]): A dictionary containing the 'test' and 'control' dataframes.
            target_field (str): The name of the target field.

        Returns:
            result (Dict[str, float]): A dictionary containing the difference between the target field values of the 'test' and 'control' dataframes.
        """
        result = {}
        if self.calc_difference_method in {"all", "ate"}:
            result["ate"] = (
                splitted_data["test"][target_field]
                - splitted_data["control"][target_field]
            ).mean()
        return result

    def calc_p_value(
        self, splitted_data: Dict[str, pd.DataFrame], target_field: str
    ) -> Dict[str, float]:
        """
        Calculates the p-value for a given data set.

        Args:
            splitted_data (Dict[str, pd.DataFrame]): A dictionary containing the split data, where the keys are 'test' and 'control' and the values are pandas DataFrames.
            target_field (str): The name of the target field.
        Returns:
            Dict[str, float]: A dictionary containing the calculated p-values, where the keys are 't_test' and 'mann_whitney' and the values are the corresponding p-values.
        """
        result = {}
        if self.calc_p_value_method in {"all", "t_test"}:
            result["t_test"] = ttest_ind(
                splitted_data["test"][target_field],
                splitted_data["control"][target_field],
            ).pvalue
        if self.calc_p_value_method in {"all", "mann_whitney"}:
            result["mann_whitney"] = mannwhitneyu(
                splitted_data["test"][target_field],
                splitted_data["control"][target_field],
            ).pvalue
        return result

    def execute(
        self, data: pd.DataFrame, target_field: str, group_field: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Executes the function by splitting the input data based on the group field and calculating the size, difference, and p-value.

        Parameters:
            data (pd.DataFrame): The input data as a pandas DataFrame.
            target_field (str): The target field to be analyzed.
            group_field (str): The field used to split the data into groups.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing the size, difference, and p-value of the split data.
                - 'size': A dictionary with the sizes of the test and control groups.
                - 'difference': A dictionary with the calculated differences between the groups.
                - 'p_value': A dictionary with the calculated p-values for each group.
        """
        splitted_data = self.split_ab(data, group_field)
        return {
            "size": {
                "test": len(splitted_data["test"]),
                "control": len(splitted_data["control"]),
            },
            "difference": self.calc_difference(splitted_data, target_field),
            "p_value": self.calc_p_value(splitted_data, target_field),
        }
