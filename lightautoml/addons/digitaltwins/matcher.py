"""Base Matcher class."""

import pandas as pd
import numpy as np
import logging
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

logger = logging.getLogger("digitaltwins")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class Matcher:
    def __init__(
            self,
            input_data,
            outcome,
            treatment,
            outcome_type="numeric",
            group_col=None,
            info_col=None,
            generate_report=GENERATE_REPORT,
            report_feat_select_dir=REPORT_FEAT_SELECT_DIR,
            timeout=TIMEOUT,
            n_threads=N_THREADS,
            n_folds=N_FOLDS,
            verbose=VERBOSE,
            use_algos=None,
            same_target_threshold=SAME_TARGET_THRESHOLD,
            interquartile_coeff=OUT_INTER_COEFF,
            drop_outliers_by_percentile=OUT_MODE_PERCENT,
            min_percentile=OUT_MIN_PERCENT,
            max_percentile=OUT_MAX_PERCENT,
            n_neighbors=10,
    ):
        """

        Args:
            input_data - input dataframe: pd.DataFrame
            outcome - target column: str
            treatment - column determine control and test groups: str
            outcome_type - values type of target column: str
            group_col - column for grouping: str
            info_col - columns with id, date or metadata, not take part in calculations: list
            generate_report - flag to create report: bool
            report_feat_select_dir - folder for report files: str
            timeout - limit work time of code: int
            n_threads - maximum number of threads: int
            n_folds - number of folds for cross-validation: int
            verbose - flag to show process stages: bool
            use_algos - list of names of LAMA algorithms for feature selection: list or str
            same_target_threshold - threshold for correlation coefficient filter (Spearman): float or Series
            interquartile_coeff - percent for drop outliers: float
            drop_outliers_by_percentile - flag to drop outliers by custom percentiles: bool
            min_percentile - minimum percentile to drop outliers: float
            max_percentile - maximum percentile to drop outliers: float
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

    def _preprocessing_data(self):
        """Turns categorical features into dummy.
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
                self.input_data = pd.get_dummies(self.input_data.drop(columns=[self.group_col] + self.info_col),
                                                 drop_first=True)
            else:
                self.input_data = pd.get_dummies(self.input_data.drop(columns=self.group_col), drop_first=True)
            self.input_data = pd.concat([self.input_data, group_col], axis=1)
            logger.debug('Categorical grouped features turned into dummy')
        if self.info_col is not None:
            self.input_data = pd.concat([self.input_data, info_col], axis=1)

    def _spearman_filter(self):
        """Applies filter by columns by correlation with outcome column
        """
        logger.info("Applying filter by spearman test - drop columns correlated with outcome")
        same_filter = SpearmanFilter(
            outcome=self.outcome, treatment=self.treatment, threshold=self.same_target_threshold
        )

        self.input_data = same_filter.perform_filter(self.input_data)

    def _outliers_filter(self):
        """Deletes outliers

        If drop_outliers_by_percentile is true, leaves values between min and max
        percentiles;
        If not, leaves only values between 25 and 75 percentile

        """
        logger.info("Applying filter of outliers")
        out_filter = OutliersFilter(
            interquartile_coeff=self.interquartile_coeff,
            mode_percentile=self.mode_percentile,
            min_percentile=self.min_percentile,
            max_percentile=self.max_percentile,
        )

        rows_for_del = out_filter.perform_filter(self.input_data)
        self.input_data = self.input_data.drop(rows_for_del, axis=0)

    def lama_feature_select(self) -> pd.DataFrame:
        """Counts feature importance
        """
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
                {'Feature': self.group_col, 'Importance': features.Importance.max()},
                ignore_index=True)
        return self.features_importance.sort_values("Importance", ascending=False)

    def _matching(self) -> tuple:
        """Realize matching

        Take in count presence of groups

        Returns:
            Results of matching and metrics

        """
        self.matcher = FaissMatcher(
            self.input_data,
            self.outcome,
            self.treatment,
            info_col=self.info_col,
            features=self.features_importance,
            group_col=self.group_col,
            n_neighbors=self.n_neighbors
        )
        logger.info("Applying matching")
        self.results = self.matcher.match()

        self.quality_result = self.matcher.matching_quality()

        return self.results, self.quality_result

    def validate_result(self, refuter='random_feature', n_sim=10, fraction=0.8):
        """Validates estimated ATE

        Validates estimated effect:
                                    1) by replacing real treatment with random placebo treatment.
                                     Estimated effect must be droped to zero, p-val < 0.05;
                                    2) by adding random feature ('random_feature'). Estimated effect shouldn't change
                                    sagnificantly, p-val > 0.05;
                                    3) estimates effect on subset of data (default fraction is 0.8). Estimated effect
                                    shouldn't change sagnificantly, p-val > 0.05.

        Args:
            refuter - refuter type ('random_treatment', 'random_feature', 'subset_refuter')
            n_sim - number of simulations: int
            fraction - subsret fraction for subset refuter only: float

        Returns:
            self.pval_dict - dict of outcome_name: (mean_effect on validation, p-value): dict

        """
        logger.info("Applying validation of result")

        self.val_dict = {k: [] for k in [self.outcome]}
        self.pval_dict = dict()

        for i in range(n_sim):
            if refuter in ["random_treatment", "random_feature"]:
                if refuter == "random_treatment":
                    self.input_data, orig_treatment, self.validate = random_treatment(self.input_data, self.treatment)
                elif refuter == "random_feature":
                    self.input_data, self.validate = random_feature(self.input_data)
                    if self.features_importance is not None and i == 0:
                        self.features_importance.append("random_feature")

                self.matcher = FaissMatcher(self.input_data, self.outcome, self.treatment, info_col=self.info_col,
                                            features=self.features_importance,
                                            group_col=self.group_col, validation=self.validate,
                                            n_neighbors=self.n_neighbors)
            elif refuter == "subset_refuter":
                df, self.validate = subset_refuter(self.input_data, self.treatment, fraction)
                self.matcher = FaissMatcher(df, self.outcome, self.treatment, info_col=self.info_col,
                                            features=self.features_importance,
                                            group_col=self.group_col, validation=self.validate,
                                            n_neighbors=self.n_neighbors)
            else:
                logger.info("Incorrect refuter name!")
                raise NameError("Incorrect refuter name!")

            if self.group_col is None:
                sim = self.matcher.match()
            else:
                sim = self.matcher.group_match()

            for key in self.val_dict.keys():
                self.val_dict[key].append(sim[key][0])

        for outcome in [self.outcome]:
            self.pval_dict.update({outcome: [np.mean(self.val_dict[outcome])]})
            self.pval_dict[outcome].append(test_significance(self.results.loc['ATE']['effect_size'],
                                                             self.val_dict[outcome]))
        if refuter == "random_treatment":
            self.input_data[self.treatment] = orig_treatment
        elif refuter == "random_feature":
            self.input_data = self.input_data.drop(columns="random_feature")
            if self.features_importance is not None:
                self.features_importance.remove("random_feature")

        return self.pval_dict

    def estimate(self, features: list = None):
        """Applies filters and outliers, then matches

        Args:
            features - list or feature_importances: list

        Returns:
            Matched df and ATE: tuple

        """
        if features is not None:
            self.features_importance = features
        return self._matching()
