"""Base Matcher class."""
import logging
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .algorithms.faiss_matcher import FaissMatcher
from .selectors.lama_feature_selector import LamaFeatureSelector
from .selectors.outliers_filter import OutliersFilter
from .selectors.spearman_filter import SpearmanFilter
from .utils.validators import random_treatment, random_feature, subset_refuter, test_significance

REPORT_FEAT_SELECT_DIR = "report_feature_selector"
REPORT_PROP_MATCHER_DIR = "report_matcher"
NAME_REPORT = "lama_interactive_report.html"
N_THREADS = 1
N_FOLDS = 4
RANDOM_STATE = 123
TEST_SIZE = 0.2
TIMEOUT = 600
VERBOSE = 2
USE_ALGOS = ["lgb"]
PROP_SCORES_COLUMN = "prop_scores"
GENERATE_REPORT = True
SAME_TARGET_THRESHOLD = 0.7
OUT_INTER_COEFF = 1.5
OUT_MODE_PERCENT = True
OUT_MIN_PERCENT = 0.02
OUT_MAX_PERCENT = 0.98

logger = logging.getLogger("hypex")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class Matcher:
    """Main class.
    """

    def __init__(
            self,
            input_data: pd.DataFrame,
            outcome: str,
            treatment: str,
            outcome_type: str = "numeric",
            group_col: str = None,
            info_col: list = None,
            generate_report: bool = GENERATE_REPORT,
            report_feat_select_dir: str = REPORT_FEAT_SELECT_DIR,
            timeout: int = TIMEOUT,
            n_threads: int = N_THREADS,
            n_folds: int = N_FOLDS,
            verbose: bool = VERBOSE,
            use_algos: list = None,
            same_target_threshold: float = SAME_TARGET_THRESHOLD,
            interquartile_coeff: float = OUT_INTER_COEFF,
            drop_outliers_by_percentile: bool = OUT_MODE_PERCENT,
            min_percentile: float = OUT_MIN_PERCENT,
            max_percentile: float = OUT_MAX_PERCENT,
            n_neighbors: int = 10,
            silent: bool = True,
            pbar: bool = True,
    ):
        """Initialize the Matcher object.

        Args:
            input_data: pd.DataFrame
                Input dataframe
            outcome: str
                Target column
            treatment: str
                Column determine control and test groups
            outcome_type: str, optional
                Values type of target column. Defaults to "numeric"
            group_col: str, optional
                Column for grouping. Defaults to None.
            info_col: list, optional
                Columns with id, date or metadata, not taking part in calculations. Defaults to None
            generate_report: bool, optional
                Flag to create report. Defaults to True
            report_feat_select_dir: str, optional
                Folder for report files. Defaults to "report_feature_selector"
            timeout: int, optional
                Limit work time of code LAMA. Defaults to 600
            n_threads: int, optional
                Maximum number of threads. Defau;ts to 1
            n_folds: int, optional
                Number of folds for cross-validation. Defaults to 4
            verbose: int, optional
                Flag to show process stages. Defaults to 2
            use_algos: list, optional
                List of names of LAMA algorithms for feature selection. Defaults to ["lgb"]
            same_target_threshold: float, optional
                Threshold for correlation coefficient filter (Spearman). Default to 0.7
            interquartile_coeff: float, optional
                Percent for drop outliers. Default to 1.5
            drop_outliers_by_percentile: bool, optional
                Flag to drop outliers by custom percentiles. Defaults to True
            min_percentile: float, optional
                Minimum percentile to drop outliers. Defaults to 0.02
            max_percentile: float, optional
                Maximum percentile to drop outliers. Defaults to 0.98
            n_neighbors: int, optional
                Number of neighbors to match. Defaults to 10
            silent: bool, optional
                Write logs in debug mode
            pbar: bool, optional
                Display progress bar while get index
        """
        if use_algos is None:
            use_algos = USE_ALGOS
        self.input_data = input_data
        self.outcome = outcome
        self.treatment = treatment
        self.group_col = group_col
        self.outcome_type = outcome_type
        self.generate_report = generate_report
        self.report_feat_select_dir = report_feat_select_dir
        self.timeout = timeout
        self.n_threads = n_threads
        self.n_folds = n_folds
        self.verbose = verbose
        self.use_algos = use_algos
        self.same_target_threshold = same_target_threshold
        self.interquartile_coeff = interquartile_coeff
        self.mode_percentile = drop_outliers_by_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.info_col = info_col
        self.features_importance = None
        self._preprocessing_data()
        self.matcher = None
        self.val_dict = None
        self.pval_dict = None
        self.new_treatment = None
        self.validate = None
        self.n_neighbors = n_neighbors
        self.silent = silent
        self.pbar = pbar

    def _preprocessing_data(self):
        """Converts categorical features into dummy variables.
        """
        if self.info_col is not None:
            info_col = self.input_data[self.info_col]

        if self.group_col is None:
            if self.info_col is not None:
                self.input_data = pd.get_dummies(self.input_data.drop(columns=self.info_col), drop_first=True)
            else:
                self.input_data = pd.get_dummies(self.input_data, drop_first=True)
            logger.debug("Categorical features turned into dummy")
        else:
            group_col = self.input_data[[self.group_col]]
            if self.info_col is not None:
                self.input_data = pd.get_dummies(
                    self.input_data.drop(columns=[self.group_col] + self.info_col), drop_first=True
                )
            else:
                self.input_data = pd.get_dummies(self.input_data.drop(columns=self.group_col), drop_first=True)
            self.input_data = pd.concat([self.input_data, group_col], axis=1)
            logger.debug("Categorical grouped features turned into dummy")
        if self.info_col is not None:
            self.input_data = pd.concat([self.input_data, info_col], axis=1)

    def _spearman_filter(self):
        """Applies a filter by dropping columns correlated with the outcome column.

        This method uses the Spearman filter to eliminate features from the dataset
        that are highly correlated with the outcome columns, based on a pre-set threshold
        """
        if self.silent:
            logger.debug("Applying filter by spearman test - drop columns correlated with outcome")
        else:
            logger.info("Applying filter by spearman test - drop columns correlated with outcome")

        same_filter = SpearmanFilter(
            outcome=self.outcome, treatment=self.treatment, threshold=self.same_target_threshold
        )

        self.input_data = same_filter.perform_filter(self.input_data)

    def _outliers_filter(self):
        """Removes outlier values from the dataset.

        This method employs an OutliersFilter. If `drop_outliers_by_percentile` is True,
        it retains only the values between the min and max percentiles
        If `drop_outliers_by_percentile` is False, it retains only the values between 2nd and 98th percentiles
        """
        if self.silent:
            logger.debug(
                f"Applying filter of outliers\ninterquartile_coeff={self.interquartile_coeff}\nmode_percentile={self.mode_percentile}\nmin_percentile={self.min_percentile}\nmax_percentile={self.max_percentile}"
            )
        else:
            logger.info(
                f"Applying filter of outliers\ninterquartile_coeff={self.interquartile_coeff}\nmode_percentile={self.mode_percentile}\nmin_percentile={self.min_percentile}\nmax_percentile={self.max_percentile}"
            )

        out_filter = OutliersFilter(
            interquartile_coeff=self.interquartile_coeff,
            mode_percentile=self.mode_percentile,
            min_percentile=self.min_percentile,
            max_percentile=self.max_percentile,
        )

        rows_for_del = out_filter.perform_filter(self.input_data)
        self.input_data = self.input_data.drop(rows_for_del, axis=0)

    def lama_feature_select(self) -> pd.DataFrame:
        """Calculates the importance of each feature.

        This method use LamaFeatureSelector to rank the importance of each feature in the dataset
        The features are then sorted by their importance with the most important feature first

        Returns:
            pd.DataFrame
                The feature importances, sorted in descending order
        """
        if self.silent:
            logger.debug("Counting feature importance")
        else:
            logger.info("Counting feature importance")

        feat_select = LamaFeatureSelector(
            outcome=self.outcome,
            outcome_type=self.outcome_type,
            treatment=self.treatment,
            timeout=self.timeout,
            n_threads=self.n_threads,
            n_folds=self.n_folds,
            verbose=self.verbose,
            generate_report=self.generate_report,
            report_dir=self.report_feat_select_dir,
            use_algos=self.use_algos,
        )
        df = self.input_data if self.group_col is None else self.input_data.drop(columns=self.group_col)

        if self.info_col is not None:
            df = df.drop(columns=self.info_col)

        features = feat_select.perform_selection(df=df)
        if self.group_col is None:
            self.features_importance = features
        else:
            self.features_importance = features.append(
                {"Feature": self.group_col, "Importance": features.Importance.max()}, ignore_index=True
            )
        return self.features_importance.sort_values("Importance", ascending=False)

    def _matching(self) -> tuple:
        """Performs matching considering the presence of groups.

        Returns:
            tuple: Results of matching and matching quality metrics

        """
        self.matcher = FaissMatcher(
            self.input_data,
            self.outcome,
            self.treatment,
            info_col=self.info_col,
            features=self.features_importance,
            group_col=self.group_col,
            n_neighbors=self.n_neighbors,
            silent=self.silent,
            pbar=self.pbar,
        )
        if self.silent:
            logger.debug("Applying matching")
        else:
            logger.info("Applying matching")

        self.results, df_matched = self.matcher.match()

        self.quality_result = self.matcher.matching_quality(df_matched)

        return self.results, self.quality_result, df_matched

    def validate_result(self, refuter: str = "random_feature", n_sim: int = 10, fraction: float = 0.8) -> dict:
        """Validates estimated ATE (Average Treatment Effect).

        Validates estimated effect:
                                    1) by replacing real treatment with random placebo treatment.
                                     Estimated effect must be droped to zero, p-val < 0.05;
                                    2) by adding random feature (`random_feature`). Estimated effect shouldn't change
                                    significantly, p-val > 0.05;
                                    3) estimates effect on subset of data (default fraction is 0.8). Estimated effect
                                    shouldn't change significantly, p-val > 0.05.

        Args:
            refuter: str
                Refuter type (`random_treatment`, `random_feature`, `subset_refuter`)
            n_sim: int
                Number of simulations
            fraction: float
                Subset fraction for subset refuter only

        Returns:
            dict: Dictionary of outcome_name: (mean_effect on validation, p-value)
        """
        if self.silent:
            logger.debug("Applying validation of result")
        else:
            logger.info("Applying validation of result")

        self.val_dict = {k: [] for k in [self.outcome]}
        self.pval_dict = dict()

        for i in tqdm(range(n_sim)):
            if refuter in ["random_treatment", "random_feature"]:
                if refuter == "random_treatment":
                    self.input_data, orig_treatment, self.validate = random_treatment(self.input_data, self.treatment)
                elif refuter == "random_feature":
                    self.input_data, self.validate = random_feature(self.input_data)
                    if self.features_importance is not None and i == 0:
                        self.features_importance.append("random_feature")

                self.matcher = FaissMatcher(
                    self.input_data,
                    self.outcome,
                    self.treatment,
                    info_col=self.info_col,
                    features=self.features_importance,
                    group_col=self.group_col,
                    validation=self.validate,
                    n_neighbors=self.n_neighbors,
                    pbar=False,
                )
            elif refuter == "subset_refuter":
                df, self.validate = subset_refuter(self.input_data, self.treatment, fraction)
                self.matcher = FaissMatcher(
                    df,
                    self.outcome,
                    self.treatment,
                    info_col=self.info_col,
                    features=self.features_importance,
                    group_col=self.group_col,
                    validation=self.validate,
                    n_neighbors=self.n_neighbors,
                    pbar=False,
                )
            else:
                logger.error("Incorrect refuter name")
                raise NameError(
                    "Incorrect refuter name! Available refuters: `random_feature`, `random_treatment`, `subset_refuter`"
                )

            if self.group_col is None:
                sim = self.matcher.match()
            else:
                sim = self.matcher.group_match()

            for key in self.val_dict.keys():
                self.val_dict[key].append(sim[key][0])

        for outcome in [self.outcome]:
            self.pval_dict.update({outcome: [np.mean(self.val_dict[outcome])]})
            self.pval_dict[outcome].append(
                test_significance(self.results.loc["ATE"]["effect_size"], self.val_dict[outcome])
            )
        if refuter == "random_treatment":
            self.input_data[self.treatment] = orig_treatment
        elif refuter == "random_feature":
            self.input_data = self.input_data.drop(columns="random_feature")
            if self.features_importance is not None:
                self.features_importance.remove("random_feature")

        return self.pval_dict

    def estimate(self, features: list = None) -> tuple:
        """Applies filters and outliers, then performs matching.

        Args:
            features: list
                Type List or feature_importances from LAMA

        Returns:
            tuple: Results of matching and matching quality metrics
        """
        if features is not None:
            self.features_importance = features
        return self._matching()

    def save(self, filename):
        """Save the object to a file using pickle.

        This method serializes the object and writes it to a file

        Args:
            filename: str
                The name of the file to write to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load an object from a file.

        This method reads a file and deserializes the object from it

        Args:
            filename: str
                The name of the file to read from.

        Returns:
            object:
                The deserialized object
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
