"""Base Matcher class."""

import pandas as pd
import numpy as np
import logging
from .algorithms.faiss_matcher import FaissMatcher
from .selectors.lama_feature_selector import LamaFeatureSelector
from .selectors.outliers_filter import OutliersFilter
from .selectors.spearman_filter import SpearmanFilter

REPORT_FEAT_SELECT_DIR = 'report_feature_selector'
REPORT_PROP_SCORE_DIR = 'report_prop_score_estimator'
REPORT_PROP_MATCHER_DIR = 'report_matcher'
NAME_REPORT = 'lama_interactive_report.html'
N_THREADS = 1
N_FOLDS = 4
RANDOM_STATE = 123
TEST_SIZE = 0.2
TIMEOUT = 600
VERBOSE = 2
USE_ALGOS = ['lgb']
PROP_SCORES_COLUMN = 'prop_scores'
GENERATE_REPORT = True
SAME_TARGET_THRESHOLD = .7
OUT_INTER_COEFF = 1.5
OUT_MODE_PERCENT = True
OUT_MIN_PERCENT = .02
OUT_MAX_PERCENT = .98

logger = logging.getLogger('matcher')
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format='[%(asctime)s | %(name)s | %(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.INFO
)

class Matcher:
    def __init__(
            self,
            df,
            outcome,
            treatment,
            outcome_type='numeric',
            is_spearman_filter=False,
            is_outliers_filter=False,
            is_feature_select=True,
            generate_report=GENERATE_REPORT,
            report_feat_select_dir=REPORT_FEAT_SELECT_DIR,
            report_prop_score_dir=REPORT_PROP_SCORE_DIR,
            report_matcher_dir=REPORT_PROP_MATCHER_DIR,
            timeout=TIMEOUT,
            n_threads=N_THREADS,
            n_folds=N_FOLDS,
            verbose=VERBOSE,
            use_algos=None,
            same_target_threshold=SAME_TARGET_THRESHOLD,
            interquartile_coeff=OUT_INTER_COEFF,
            mode_percentile=OUT_MODE_PERCENT,
            min_percentile=OUT_MIN_PERCENT,
            max_percentile=OUT_MAX_PERCENT, group_col=None, quality_check=False
    ):
        if use_algos is None:
            use_algos = USE_ALGOS
        self.df = df
        self.data = self.df.copy().drop([outcome], axis=1)
        self.outcome = outcome
        self.treatment = treatment
        self.group_col = group_col
        self.outcome_type = outcome_type
        self.is_spearman_filter = is_spearman_filter
        self.is_outliers_filter = is_outliers_filter
        self.is_feature_select = is_feature_select
        self.generate_report = generate_report
        self.report_feat_select_dir = report_feat_select_dir
        self.report_prop_score_dir = report_prop_score_dir
        self.report_matcher_dir = report_matcher_dir
        self.timeout = timeout
        self.n_threads = n_threads
        self.n_folds = n_folds
        self.verbose = verbose
        self.use_algos = use_algos
        self.same_target_threshold = same_target_threshold
        self.interquartile_coeff = interquartile_coeff
        self.mode_percentile = mode_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.features = None
        self._preprocessing_data()
        self.quality_check = quality_check
        self.matcher = None
        self.val_dict = {k: [] for k in [self.outcome]}
        self.pval_dict = None
        self.new_treatment = None
        self.validate = None

    def _preprocessing_data(self):
        """Turns categorical features into dummy.
        """
        if self.group_col is None:
            self.df = pd.get_dummies(self.df, drop_first=True)
            logger.debug('Categorical features turned into dummy')
        else:
            group_col = self.df[[self.group_col]]
            self.df = pd.get_dummies(self.df.drop(columns=self.group_col), drop_first=True)
            self.df = pd.concat([self.df, group_col], axis=1)
            logger.debug('Categorical grouped features turned into dummy')

    def _spearman_filter(self):
        """Applies filter by columns by correlation with outcome column
        """
        logger.info('Applying filter by spearman test - drop columns correlated with outcome')
        same_filter = SpearmanFilter(
            outcome=self.outcome,
            treatment=self.treatment,
            threshold=self.same_target_threshold
        )

        self.df = same_filter.perform_filter(self.df)


    def _outliers_filter(self):
        """Deletes outliers

        If mode_percentile is true, leaves values between min and max
        percentiles;
        If not, leaves only values between 25 and 75 percentile

        """
        logger.info('Applying filter of outliers')
        out_filter = OutliersFilter(
            interquartile_coeff=self.interquartile_coeff,
            mode_percentile=self.mode_percentile,
            min_percentile=self.min_percentile,
            max_percentile=self.max_percentile
        )

        rows_for_del = out_filter.perform_filter(self.df)
        self.df = self.df.drop(rows_for_del, axis=0)
        self.data = self.data.drop(rows_for_del, axis=0)

    def _feature_select(self):
        """Counts feature importance
        """
        logger.info('Counting feature importance')
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
            use_algos=self.use_algos
        )
        if self.group_col is None:
            features = feat_select.perform_selection(df=self.df)
        else:
            logger.info('Feature importance counted without group columns')
            features = feat_select.perform_selection(df=self.df.drop(columns=self.group_col))
        self.features = features

    def _matching(self):
        """Realize matching

        Take in count presence of groups

        Returns:
            Tuple of matched df and ATE

        """
        self.matcher = FaissMatcher(self.df, self.data, self.outcome, self.treatment, self.features,
                                    group_col=self.group_col)
        if self.group_col is None:
            logger.info('Applying matching')
            df_matched, ate = self.matcher.match()
        else:
            logger.info('Applying group matching')
            df_matched, ate = self.matcher.group_match()

        if self.quality_check:
            logger.info('Checking quality')
            self.quality_result = self.matcher.matching_quality()

        return df_matched, ate

    def validate_result(self, n_sim=10):
        """Validates estimated effect

        Validates estimated effect by replacing real treatment with random
        placebo treatment.
        Estimated effect must be droped to zero

        Args:
            n_sim - number of simulations: int

        Returns:
            self.pval_dict - dict of p-values: dict

        """
        logger.info('Applying validation of result')
        for i in range(n_sim):
            prop1 = self.df[self.treatment].sum() / self.df.shape[0]
            prop0 = 1 - prop1
            self.new_treatment = np.random.choice([0, 1], size=self.df.shape[0], p=[prop0, prop1])
            self.validate = 1
            self.df = self.df.drop(columns=self.treatment)
            self.df[self.treatment] = self.new_treatment
            self.data = self.data.drop(columns=self.treatment)
            self.data[self.treatment] = self.new_treatment
            self.matcher = FaissMatcher(self.df, self.data, self.outcome, self.treatment, self.features,
                                        group_col=self.group_col, validation=self.validate)
            if self.group_col is None:
                sim = self.matcher.match()
            else:
                sim = self.matcher.group_match()
            for key in self.val_dict.keys():
                self.val_dict[key].append(sim[key][0])
        self.pval_dict = dict()
        for outcome in [self.outcome]:
            self.pval_dict.update({outcome: np.mean(self.val_dict[outcome])})
        return self.pval_dict

    def estimate(self):
        """Applies filters and outliers, then matches

        Returns:
            Tuple of matched df and ATE

        """
        if self.is_spearman_filter:
            self._spearman_filter()
        if self.is_outliers_filter:
            self._outliers_filter()
        if self.is_feature_select:
            self._feature_select()

        return self._matching()
