from Matcher.selectors.OutliersFilter import OutliersFilter
from Matcher.selectors.SpearmanFilter import SpearmanFilter
from Matcher.selectors.LamaFeatureSelector import LamaFeatureSelector
from Matcher.algorithms.FaissMatcher import FaissMatcher
import pandas as pd



REPORT_FEAT_SELECT_DIR = 'report_feature_selector'
REPORT_PROP_SCORE_DIR = 'report_prop_score_estimator'
REPORT_PROP_MATCHER_DIR = 'report_matcher'
NAME_REPORT = 'lama_interactive_report.html'
N_THREADS = 32
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


class Matching:
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
            max_percentile=OUT_MAX_PERCENT,
    ):
        if use_algos is None:
            use_algos = USE_ALGOS
        self.df = df
        self.data = self.df.copy().drop([outcome], axis=1)
        self.outcome = outcome
        self.treatment = treatment
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


    def _preprocessing_data(self):
        self.df = pd.get_dummies(self.df, drop_first=True)


    def _spearman_filter(self):
        same_filter = SpearmanFilter(
            outcome=self.outcome,
            treatment=self.treatment,
            threshold=self.same_target_threshold
        )

        self.df = same_filter.perform_filter(self.df)
        self.data = self.data.merge(self.df, how='left',
                                    on=list(self.df.columns).remove(self.outcome))\
            .drop([self.outcome], axis=1)

    def _outliers_filter(self):
        out_filter = OutliersFilter(
            interquartile_coeff=self.interquartile_coeff,
            mode_percentile=self.mode_percentile,
            min_percentile=self.min_percentile,
            max_percentile=self.max_percentile
        )

        self.df = out_filter.perform_filter(self.df)
        self.data = self.data.merge(self.df, how='right',
                                    on=list(self.df.columns).remove(self.outcome)) \
            .drop([self.outcome], axis=1)


    def _feature_select(self):
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

        features = feat_select.perform_selection(df=self.df)
        self.features = features

    def _matching(self):
        matcher  = FaissMatcher(self.df, self.data, self.outcome, self.treatment, self.features)
        df_matched, ate = matcher.match()
        return df_matched, ate

    def estimate(self):
        if self.is_spearman_filter:
            self._spearman_filter()
        if self.is_outliers_filter:
            self._outliers_filter()
        if self.is_feature_select:
            self._feature_select()

        return self._matching()

