"""Base Matcher class."""
import logging
import pickle

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from .algorithms.faiss_matcher import FaissMatcher
from .algorithms.no_replacement_matching import MatcherNoReplacement
from .selectors.lama_feature_selector import LamaFeatureSelector
from .selectors.spearman_filter import SpearmanFilter
from .selectors.outliers_filter import OutliersFilter
from .selectors.base_filtration import const_filtration, nan_filtration
from .utils.validators import random_feature
from .utils.validators import random_treatment
from .utils.validators import subset_refuter
from .utils.validators import test_significance


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
    """Class for compile full pipeline of Matching in Causal Inference task.

    Matcher steps:
        - Read, analyze data
        - Feature selection via LightAutoML
        - Converting a dataset with features to another space via Cholesky decomposition
          In the new space, the distance L2 becomes equivalent to the Mahalanobis distance.
          This allows us to use faiss to search for nearest objects, which can search only by L2 metric,
          but without violating the methodology of matching,
          for which it is important to count by the Mahalanobis distance
        - Finding the nearest neighbors for each unit (with duplicates) using faiss.
          For each of the control group, neighbors from the target group are matched and vice versa.
        - Calculation bias
        - Creating matched df (Wide df with pairs)
        - Calculation metrics: ATE, ATT, ATC, p-value,  and сonfidence intervals
        - Calculation quality: PS-test, KS test, SMD test
        - Returns metrics as dataframe, quality results as dict of df's and df_matched
        - After receiving the result, the result should be validated using :func:`~lightautoml.addons.hypex.matcher.Matcher.validate_result`

    Example:
        Common usecase - base pipeline for matching

        >>> # Base info
        >>> treatment = "treatment" # Column name with info about 'treatment' 0 or 1
        >>> target = "target" # Column name with target
        >>>
        >>> # Optional
        >>> info_col = ["user_id", 'address'] # Columns that will not participate in the match and are informative.
        >>> group_col = "CatCol" # Column name for strict comparison (for a categorical feature)
        >>>
        >>> # Matching
        >>> model = Matcher(data, outcome=target, treatment=treatment, info_col=info_col, group_col=group_col)
        >>> features = model.lama_feature_select() # Feature selection via lama
        >>> results, quality, df_matched = model.estimate(features=some_features) # Performs matching
        >>>
        >>> model.validate_result()
    """

    def __init__(
        self,
        input_data: pd.DataFrame,
        treatment: str,
        outcome: str = None,
        outcome_type: str = "numeric",
        group_col: str = None,
        info_col: list = None,
        weights: dict = None,
        base_filtration: bool = False,
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
        n_neighbors: int = 1,
        silent: bool = True,
        pbar: bool = True,
    ):
        """Initialize the Matcher object.

        Args:
            input_data:
                Input dataframe
            outcome:
                Target column
            treatment:
                Column determine control and test groups
            outcome_type:
                Values type of target column. Defaults to "numeric"
            group_col:
                Column for grouping. Defaults to None.
            info_col:
                Columns with id, date or metadata, not taking part in calculations. Defaults to None
            weights:
                weights for numeric columns in order to increase matching quality by weighted feature.
                By default, is None (all features have the same weight equal to 1). Example: {'feature_1': 10}
            base_filtration:
                To use or not base filtration of features in order to remove all constant or almost all constant, bool.
                Default is False.
            generate_report:
                Flag to create report. Defaults to True
            report_feat_select_dir:
                Folder for report files. Defaults to "report_feature_selector"
            timeout:
                Limit work time of code LAMA. Defaults to 600
            n_threads:
                Maximum number of threads. Defaults to 1
            n_folds:
                Number of folds for cross-validation. Defaults to 4
            verbose:
                Flag to show process stages. Defaults to 2
            use_algos:
                List of names of LAMA algorithms for feature selection. Defaults to ["lgb"]
            same_target_threshold:
                Threshold for correlation coefficient filter (Spearman). Default to 0.7
            interquartile_coeff:
                Percent for drop outliers. Default to 1.5
            drop_outliers_by_percentile:
                Flag to drop outliers by custom percentiles. Defaults to True
            min_percentile:
                Minimum percentile to drop outliers. Defaults to 0.02
            max_percentile:
                Maximum percentile to drop outliers. Defaults to 0.98
            n_neighbors:
                Number of neighbors to match (in fact you may see more then n matches as every match may have more then
                one neighbor with the same distance). Default value is 1.
            silent:
                Write logs in debug mode
            pbar:
                Display progress bar while get index
        """
        if use_algos is None:
            use_algos = USE_ALGOS
        self.input_data = input_data
        if outcome is None:
            outcome = list()
        self.outcomes = outcome if type(outcome) == list else [outcome]
        self.treatment = treatment
        self.group_col = group_col
        self.info_col = info_col
        self.outcome_type = outcome_type
        self.weights = weights
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
        self.base_filtration = base_filtration
        self.features_importance = None
        self.matcher = None
        self.val_dict = None
        self.pval_dict = None
        self.new_treatment = None
        self.validate = None
        self.dropped_features = []
        self.n_neighbors = n_neighbors
        self.silent = silent
        self.pbar = pbar
        self._preprocessing_data()

    def _convert_categorical_to_dummy(self):
        """Converts categorical variables to dummy variables.

        Returns:
            Data with categorical variables converted to dummy variables.
        """
        info_col = self.info_col if self.info_col is not None else []
        group_col = [self.group_col] if self.group_col is not None else []

        columns_to_drop = info_col + group_col
        if columns_to_drop is not None:
            data = self.input_data.drop(columns=columns_to_drop)
        else:
            data = self.input_data
        dummy_data = pd.get_dummies(data, drop_first=True)
        return dummy_data

    def _preprocessing_data(self):
        """Converts categorical features into dummy variables."""
        info_col = self.info_col if self.info_col is not None else []
        group_col = [self.group_col] if self.group_col is not None else []
        columns_to_drop = info_col + group_col + self.outcomes + [self.treatment]
        if self.base_filtration:
            filtered_features = nan_filtration(self.input_data.drop(columns=columns_to_drop))
            self.dropped_features = [f for f in self.input_data.columns if f not in filtered_features + columns_to_drop]
            self.input_data = self.input_data[filtered_features + columns_to_drop]
        nan_counts = self.input_data.isna().sum().sum()
        if nan_counts != 0:
            self._log(f"Number of NaN values filled with zeros: {nan_counts}", silent=False)
            self.input_data = self.input_data.fillna(0)

        if self.group_col is not None:
            group_col = self.input_data[[self.group_col]]
        if self.info_col is not None:
            info_col = self.input_data[self.info_col]

        self.input_data = self._convert_categorical_to_dummy()
        if self.group_col is not None:
            self.input_data = pd.concat([self.input_data, group_col], axis=1)

        if self.info_col is not None:
            self.input_data = pd.concat([self.input_data, info_col], axis=1)

        if self.base_filtration:
            filtered_features = const_filtration(self.input_data.drop(columns=columns_to_drop))
            self.dropped_features = np.concatenate(
                (
                    self.dropped_features,
                    [f for f in self.input_data.columns if f not in filtered_features + columns_to_drop],
                )
            )
            self.input_data = self.input_data[filtered_features + columns_to_drop]

        self._log("Categorical features turned into dummy")

    def _apply_filter(self, filter_class, *filter_args):
        """Applies a filter to the input data.

        Args:
            filter_class:
                The class of the filter to apply.
            *filter_args:
                Arguments to pass to the filter class.
        """
        filter_instance = filter_class(*filter_args)
        self.input_data = filter_instance.perform_filter(self.input_data)

    def _spearman_filter(self):
        """Applies a filter by dropping columns correlated with the outcome column.

        This method uses the Spearman filter to eliminate features from the dataset
        that are highly correlated with the outcome columns, based on a pre-set threshold
        """
        self._log("Applying filter by spearman test - drop columns correlated with outcome")
        self._apply_filter(SpearmanFilter, self.outcomes[0], self.treatment, self.same_target_threshold)

    def _outliers_filter(self):
        """Removes outlier values from the dataset.

        This method employs an OutliersFilter. If `drop_outliers_by_percentile` is True,
        it retains only the values between the min and max percentiles
        If `drop_outliers_by_percentile` is False, it retains only the values between 2nd and 98th percentiles
        """
        self._log(
            f"Applying filter of outliers\n"
            f"interquartile_coeff={self.interquartile_coeff}\n"
            f"mode_percentile={self.mode_percentile}\n"
            f"min_percentile={self.min_percentile}\n"
            f"max_percentile={self.max_percentile}"
        )

        self._apply_filter(
            OutliersFilter, self.interquartile_coeff, self.mode_percentile, self.min_percentile, self.max_percentile
        )

    def match_no_rep(self, threshold: float = 0.1) -> pd.DataFrame:
        """Matching groups with no replacement.

        It's done by optimizing the linear sum of
        distances between pairs of treatment and control samples.

        Args:
            threshold: caliper for minimum deviation between test and control groups. in case weights is not None.

        Returns:
              Matched dataframe with no replacements.
        """
        a = self.input_data[self.treatment]
        X = self.input_data.drop(columns=self.treatment)
        if self.info_col is not None:
            X = X.drop(columns=self.info_col)

        index_matched = MatcherNoReplacement(X, a, self.weights).match()
        filtred_matches = index_matched.loc[1].iloc[self.input_data[a == 1].index].matches[index_matched.loc[1].iloc[self.input_data[a == 1].index].matches.apply(lambda x: x != [])]

        if self.weights is not None:
            weighted_features = [f for f in self.weights.keys()]
            index_dict = dict()
            for w in weighted_features:
                source = self.input_data.loc[np.concatenate(filtred_matches.values)][w].values
                target = self.input_data.loc[filtred_matches.index.to_list()][w].values
                index = abs(source - target) <= abs(source) * threshold
                index_dict.update({w: index})
            index_filtered = sum(index_dict.values()) == len(self.weights)
            matched_data = pd.concat(
                [self.input_data.loc[filtred_matches.index.to_list()].iloc[index_filtered],
                 self.input_data.loc[np.concatenate(filtred_matches.values)].iloc[index_filtered]]
            )
        else:
            matched_data = pd.concat([self.input_data.loc[filtred_matches.index.to_list()],
                                      self.input_data.loc[np.concatenate(filtred_matches.values)]])
        return matched_data

    def lama_feature_select(self) -> pd.DataFrame:
        """Calculates the importance of each feature.

        This method use LamaFeatureSelector to rank the importance of each feature in the dataset
        The features are then sorted by their importance with the most important feature first

        Returns:
            The feature importances, sorted in descending order
        """
        self._log("Counting feature importance")

        feat_select = LamaFeatureSelector(
            outcome=self.outcomes[0],
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

    def _create_faiss_matcher(self, df=None, validation=None):
        """Creates a FaissMatcher object.

        Args:
            df:
                The dataframe to use. If None, uses self.input_data.
            validation:
                Whether to use the matcher for validation. If None, determines based on whether
        """
        if df is None:
            df = self.input_data
        self.matcher = FaissMatcher(
            df,
            self.outcomes,
            self.treatment,
            info_col=self.info_col,
            weights=self.weights,
            features=self.features_importance,
            group_col=self.group_col,
            validation=validation,
            n_neighbors=self.n_neighbors,
            pbar=False if validation else self.pbar,
        )

    def _perform_validation(self):
        """Performs validation using the FaissMatcher."""
        if self.group_col is None:
            sim = self.matcher.match()
        else:
            sim = self.matcher.group_match()
        for key in self.val_dict.keys():
            self.val_dict[key].append(sim[key][0])

    def _log(self, message, silent=None):
        """Logs a message at the appropriate level.

        Args:
            message:
                The message to log.
            silent:
                If silent, logs will be only info
        """
        if silent is None:
            silent = self.silent
        if silent:
            logger.debug(message)
        else:
            logger.info(message)

    def _matching(self) -> tuple:
        """Performs matching considering the presence of groups.

        Returns:
            Results of matching and matching quality metrics
        """
        self._create_faiss_matcher()
        self._log("Applying matching")

        self.results, df_matched = self.matcher.match()

        self.quality_result = self.matcher.matching_quality(df_matched)

        return self.results, self.quality_result, df_matched

    def validate_result(
        self, refuter: str = "random_feature", effect_type: str = "ate", n_sim: int = 10, fraction: float = 0.8
    ) -> dict:
        """Validates estimated ATE (Average Treatment Effect).

        Validates estimated effect:
                                    1) by replacing real treatment with random placebo treatment.
                                     Estimated effect must be droped to zero, p-val > 0.05;
                                    2) by adding random feature (`random_feature`). Estimated effect shouldn't change
                                    significantly, p-val < 0.05;
                                    3) estimates effect on subset of data (default fraction is 0.8). Estimated effect
                                    shouldn't change significantly, p-val < 0.05.

        Args:
            refuter:
                Refuter type (`random_treatment`, `random_feature`, `subset_refuter`)
            effect_type:
                Which effect to validate (`ate`, `att`, `atc`)
            n_sim:
                Number of simulations
            fraction:
                Subset fraction for subset refuter only

        Returns:
            Dictionary of outcome_name (mean_effect on validation, p-value)
        """
        if self.silent:
            logger.debug("Applying validation of result")
        else:
            logger.info("Applying validation of result")

        self.val_dict = {k: [] for k in self.outcomes}
        self.pval_dict = dict()

        effect_dict = {"ate": 0, "atc": 1, "att": 2}

        assert effect_type in effect_dict.keys()

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
                    self.outcomes,
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
                    self.outcomes,
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

        for outcome in self.outcomes:
            self.pval_dict.update({outcome: [np.mean(self.val_dict[outcome])]})
            self.pval_dict[outcome].append(
                test_significance(
                    self.results.query("outcome==@outcome").loc[effect_type.upper()]["effect_size"],
                    self.val_dict[outcome],
                )
            )
        if refuter == "random_treatment":
            self.input_data[self.treatment] = orig_treatment
        elif refuter == "random_feature":
            self.input_data = self.input_data.drop(columns="random_feature")
            if self.features_importance is not None:
                self.features_importance.remove("random_feature")

        return self.pval_dict

    def estimate(self, features: list = None) -> tuple:
        """Performs matching via Mahalanobis distance.

        Args:
            features:
                List or feature_importances from LAMA of features for matching

        Returns:
            Results of matching and matching quality metrics
        """
        if features is not None:
            self.features_importance = features
        return self._matching()

    def save(self, filename):
        """Save the object to a file using pickle.

        This method serializes the object and writes it to a file

        Args:
            filename:
                The name of the file to write to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load an object from a file.

        This method reads a file and deserializes the object from it

        Args:
            filename:
                The name of the file to read from.

        Returns:
                The deserialized object
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
