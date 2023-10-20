import warnings
from copy import copy

from IPython.display import display
from pathlib import Path
from sklearn.utils import shuffle
from typing import Iterable, Union, Optional, Dict, Any, Tuple

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu


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
    test_group.loc[:, "group"] = "test"
    control_group.loc[:, "group"] = "control"

    merged_data = pd.concat([test_group, control_group], ignore_index=True)

    return merged_data


class AATest:
    def __init__(
        self,
        target_fields: Union[Iterable[str], str],
        info_cols: Union[Iterable[str], str] = None,
        group_cols: Union[str, Iterable[str]] = None,
        quant_field: str = None,
        mode: str = "simple",
    ):
        self.target_fields = [target_fields] if isinstance(target_fields, str) else target_fields
        self.info_cols = [info_cols] if isinstance(info_cols, str) else info_cols
        self.group_cols = [group_cols] if isinstance(group_cols, str) else group_cols
        self.quant_field = quant_field
        self.mode = mode

    def _preprocessing_data(self, data):
        """Converts categorical variables to dummy variables.

        Returns:
            Data with categorical variables converted to dummy variables.
        """
        # categorical to dummies
        prep_data = data.copy()
        init_cols = data.columns

        dont_binarize_cols = copy(self.group_cols) or []
        if self.quant_field is not None:
            dont_binarize_cols.append(self.quant_field)

        # if self.group_cols is not None:
        prep_data = pd.get_dummies(prep_data.drop(columns=dont_binarize_cols), dummy_na=True)
        prep_data = prep_data.merge(data[dont_binarize_cols], left_index=True, right_index=True)

        # fix if dummy_na is const=0
        dummies_cols = set(prep_data.columns) - set(init_cols)
        const_columns = [col for col in dummies_cols if prep_data[col].nunique() <= 1]  # choose constant_columns

        # drop constant dummy columns and info columns
        cols_to_drop = const_columns + (self.info_cols if self.info_cols is not None else [])
        return prep_data.drop(columns=cols_to_drop)

    def __simple_mode(self, data: pd.DataFrame, random_state: int = None):
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

    def split(self, data, preprocessing_data=True, random_state: int = None) -> Dict:
        """Divides sample on two groups.

        Args:
            data: raw input data
            preprocessing_data: флаг указания на необходимость преобразования
            random_state: one integer to fix split

        Returns:
            result: dict of indexes with division on test and control group
        """
        if preprocessing_data:
            data = self._preprocessing_data(data)
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
        result["control_indexes"] = list(set(result["control_indexes"]))

        return result

    @staticmethod
    def _postprep_data(data, spit_indexes: Dict = None):
        """prep data to show user (add info_cols and decode binary variables)

        Args:
            data: raw input data
            spit_indexes: dict of indexes with separation on test and control group

        Returns:
            data: separated init data with column "group"
        """
        # prep data to show user (add info_cols and decode binary variables)
        test = data.loc[spit_indexes["test_indexes"]]
        control = data.loc[spit_indexes["control_indexes"]]
        data = merge_groups(test, control)

        return data

    @staticmethod
    def calc_ab_delta(a_mean, b_mean, mode="percentile"):
        if mode == "percentile":
            return (1 - a_mean / b_mean) * 100
        if mode == "absolute":
            return b_mean - a_mean
        if mode == "relative":
            return 1 - a_mean / b_mean

    def sampling_metrics(self, data, alpha: float = 0.05, random_state: int = None, preprocessed_data=None):
        """

        Args:
            data: raw input data
            alpha: Threshold to check statistical hypothesis; usually 0.05
            random_state: Random seeds for searching
            preprocessed_data: pre-preprocessed data

        Returns:
            result: tuple of
                1) metrics dataframe (stat tests) and
                2) dict of random state with test_control dataframe
        """

        data_from_sampling_dict = {}
        scores = []
        t_result = {"random_state": random_state}

        split = self.split(data, preprocessed_data is None, random_state)

        a = data.loc[split["test_indexes"]]
        b = data.loc[split["control_indexes"]]

        # prep data to show user (merge indexes and init data)
        data_from_sampling_dict[random_state] = self._postprep_data(data, split)

        for tf in self.target_fields:
            ta = a[tf]
            tb = b[tf]

            t_result[f"{tf} a mean"] = ta.mean()
            t_result[f"{tf} b mean"] = tb.mean()
            t_result[f"{tf} ab delta"] = self.calc_ab_delta(
                t_result[f"{tf} a mean"], t_result[f"{tf} b mean"], "absolute"
            )
            t_result[f"{tf} ab delta %"] = self.calc_ab_delta(
                t_result[f"{tf} a mean"], t_result[f"{tf} b mean"], "percentile"
            )
            t_result[f"{tf} t_test p_value"] = ttest_ind(ta, tb, nan_policy="omit").pvalue
            t_result[f"{tf} ks_test p_value"] = ks_2samp(ta, tb).pvalue
            t_result[f"{tf} t_test passed"] = t_result[f"{tf} t_test p_value"] > alpha
            t_result[f"{tf} ks_test passed"] = t_result[f"{tf} ks_test p_value"] > alpha
            scores.append((t_result[f"{tf} t_test p_value"] + t_result[f"{tf} ks_test p_value"]) / 2)

        t_result["mean_tests_score"] = np.mean(scores)
        result = {"metrics": t_result, "data_from_experiment": data_from_sampling_dict}

        return result

    def search_dist_uniform_sampling(
        self,
        data,
        alpha: float = 0.05,
        iterations: int = 10,
        file_name: Union[Path, str] = None,
        write_mode: str = "full",
        write_step: int = None,
        pbar: bool = True,
    ) -> Optional[Tuple[pd.DataFrame, Dict[Any, Dict]]]:
        """Chooses random_state for finding homogeneous distribution.

        Args:
            data: raw input data
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

        preprocessed_data = self._preprocessing_data(data)

        if write_mode not in ("full", "all", "any"):
            warnings.warn(f"Write mode '{write_mode}' is not supported. Mode 'full' will be used")
            write_mode = "full"

        for i, rs in tqdm(enumerate(random_states), total=len(random_states), disable=not pbar):
            res = self.sampling_metrics(data, alpha=alpha, random_state=rs, preprocessed_data=preprocessed_data)
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
        self, calc_difference_method: str = "all", calc_p_value_method: str = "all",
    ):
        """Initializes the ABTest class.

        Args:
            calc_difference_method:
                The method used to calculate the difference:
                    'all' [default] - all metrics
                    'ate' - basic difference in means of targets in test and control group
                    'diff_in_diff' - difference in difference value,
                                     performs pre-post analysis (required values of target before pilot)
                    'cuped' - Controlled-Experiment using Pre-Experiment Data value,
                              performs pre-post analysis (required values of target before pilot)
            calc_p_value_method:
                The method used to calculate the p-value. Defaults to 'all'
        """
        self.calc_difference_method = calc_difference_method
        self.calc_p_value_method = calc_p_value_method
        self.results = None

    @staticmethod
    def split_ab(data: pd.DataFrame, group_field: str) -> Dict[str, pd.DataFrame]:
        """Splits a pandas DataFrame into two separate dataframes based on a specified group field.

        Args:
            data:
                The input dataframe to be split
            group_field:
                The column name representing the group field

        Returns:
            splitted_data:
                A dictionary containing two dataframes, 'test' and 'control', where 'test' contains rows where the
                group field is 'test', and 'control' contains rows where the group field is 'control'.
        """
        splitted_data = {
            "test": data[data[group_field] == "test"],
            "control": data[data[group_field] == "control"],
        }
        return splitted_data

    @staticmethod
    def cuped(test_data: pd.DataFrame, control_data: pd.DataFrame, target_field: str, target_field_before: str):
        """Counts CUPED (Controlled-Experiment using Pre-Experiment Data) in absolute values.

        Metric uses pre-post analysis of target, uses to minimize variance of effect:
        ATE = mean(test_cuped) - mean(control_cuped)
            , where
            test_cuped = target__test - theta * target_before__test
            control_cuped = target__control - theta * target_before__control
                , where
                theta = (cov_test + cov_control) / (var_test + var_control)
                    , where
                    cov_test = cov(target__test, target_before__test)
                    cov_control = cov(target__control, target_before__control)
                    var_test = var(target_before__test)
                    var_control = var(target_before__control)

        Args:
            test_data:
                Input data of test group
                Should include target before and after pilot
            control_data:
                Input data of control group
                Should include target before and after pilot
            target_field:
                Column name of target after pilot
            target_field_before:
                Column name of target before pilot

        Returns:
            result:
                Named tuple with pvalue, effect, ci_length, left_bound and right_bound
        """
        control = control_data[target_field]
        control_before = control_data[target_field_before]
        test = test_data[target_field]
        test_before = test_data[target_field_before]

        theta = (np.cov(control, control_before)[0, 1] + np.cov(test, test_before)[0, 1]) / (
            np.var(control_before) + np.var(test_before)
        )

        control_cuped = control - theta * control_before
        test_cuped = test - theta * test_before

        mean_control = np.mean(control_cuped)
        mean_test = np.mean(test_cuped)

        difference_mean = mean_test - mean_control

        return difference_mean

    @staticmethod
    def diff_in_diff(test_data: pd.DataFrame, control_data: pd.DataFrame, target_field: str, target_field_before: str):
        """Counts Difference in Difference.

        Metric uses pre-post analysis and counts difference in means in data before and after pilot:
        ATE = (y_test_after - y_control_after) - (y_test_before - y_control_before)

        Args:
            test_data: input data of test group
            control_data: input data of control group
            target_field: column name of target after pilot
            target_field_before: column name of target before pilot

        Returns:
            did: value of difference in difference
        """
        mean_test = np.mean(test_data[target_field])
        mean_control = np.mean(control_data[target_field])

        mean_test_before = np.mean(test_data[target_field_before])
        mean_control_before = np.mean(control_data[target_field_before])
        did = (mean_test - mean_control) - (mean_test_before - mean_control_before)

        return did

    def calc_difference(
        self, splitted_data: Dict[str, pd.DataFrame], target_field: str, target_field_before: str = None
    ) -> Dict[str, float]:
        """Calculates the difference between the target field values of the 'test' and 'control' dataframes.

        Args:
            splitted_data:
                A dictionary containing the 'test' and 'control' dataframes
            target_field:
                The name of the target field contains data after pilot
            target_field_before:
                The name of the target field contains data before pilot

        Returns:
            result:
                A dictionary containing the difference between the target field
                values of the 'test' and 'control' dataframes
        """
        result = {}
        if self.calc_difference_method in {"all", "diff_in_diff", "cuped"}:
            if target_field_before is None:
                raise ValueError(
                    "For calculation metrics 'cuped' or 'diff_in_diff' field 'target_field_before' is required.\n"
                    "Metric 'ate'(=diff-in-means) can be used without 'target_field_before'"
                )

        if self.calc_difference_method in {"all", "ate"}:
            result["ate"] = (
                splitted_data["test"][target_field].values - splitted_data["control"][target_field].values
            ).mean()

        if self.calc_difference_method in {"all", "cuped"}:
            result["cuped"] = self.cuped(
                test_data=splitted_data["test"],
                control_data=splitted_data["control"],
                target_field=target_field,
                target_field_before=target_field_before,
            )

        if self.calc_difference_method in {"all", "diff_in_diff"}:
            result["diff_in_diff"] = self.diff_in_diff(
                test_data=splitted_data["test"],
                control_data=splitted_data["control"],
                target_field=target_field,
                target_field_before=target_field_before,
            )

        return result

    def calc_p_value(self, splitted_data: Dict[str, pd.DataFrame], target_field: str) -> Dict[str, float]:
        """Calculates the p-value for a given data set.

        Args:
            splitted_data:
                A dictionary containing the split data, where the keys are 'test' and 'control'
                and the values are pandas DataFrames
            target_field:
                The name of the target field
        Returns:
            result:
                A dictionary containing the calculated p-values, where the keys are 't_test' and 'mann_whitney'
                and the values are the corresponding p-values
        """
        result = {}
        if self.calc_p_value_method in {"all", "t_test"}:
            result["t_test"] = ttest_ind(
                splitted_data["test"][target_field], splitted_data["control"][target_field],
            ).pvalue

        if self.calc_p_value_method in {"all", "mann_whitney"}:
            result["mann_whitney"] = mannwhitneyu(
                splitted_data["test"][target_field], splitted_data["control"][target_field],
            ).pvalue

        return result

    def execute(
        self, data: pd.DataFrame, target_field: str, group_field: str, target_field_before: str = None
    ) -> Dict[str, Dict[str, float]]:
        """Splits the input data based on the group field and calculates the size, difference, and p-value.

        Parameters:
            data: The input data as a pandas DataFrame.
            target_field: The target field to be analyzed.
            group_field: The field used to split the data into groups.
            target_field_before: The target field without treatment to be analyzed

        Returns:
            results:
                A dictionary containing the size, difference, and p-value of the split data
                    'size': A dictionary with the sizes of the test and control groups
                    'difference': A dictionary with the calculated differences between the groups
                    'p_value': A dictionary with the calculated p-values for each group
        """
        splitted_data = self.split_ab(data, group_field)

        results = {
            "size": {"test": len(splitted_data["test"]), "control": len(splitted_data["control"])},
            "difference": self.calc_difference(splitted_data, target_field, target_field_before),
            "p_value": self.calc_p_value(splitted_data, target_field),
        }

        self.results = results

        return results

    def show_beautiful_result(self):
        for k in self.results.keys():
            display(pd.DataFrame(self.results[k], index=[k]).T)
