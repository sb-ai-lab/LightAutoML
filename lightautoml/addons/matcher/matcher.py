"""Base Matcher class."""

import pandas as pd
import numpy as np
import logging
from .algorithms.faiss_matcher import FaissMatcher
from .selectors.lama_feature_selector import LamaFeatureSelector
from .selectors.outliers_filter import OutliersFilter
from .selectors.spearman_filter import SpearmanFilter
from .utils.validators import random_treatment, random_feature, subset_refuter, test_significance

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
            group_col=None,
            info_col=[],
            required_col=None,
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
            max_percentile=OUT_MAX_PERCENT
    ):
        if use_algos is None:
            use_algos = USE_ALGOS
        self.df = df
        self.outcome = outcome
        self.treatment = treatment
        self.group_col = group_col
        self.required_col = required_col
        self.outcome_type = outcome_type
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
        self.info_col = info_col
        self.features_importance = None
        self._preprocessing_data()
        self.matcher = None
        self.val_dict = None
        self.pval_dict = None
        self.new_treatment = None
        self.validate = None

    def _preprocessing_data(self):
        """Turns categorical features into dummy.
        """
        if self.info_col is not None:
        	info_col = self.df[self.info_col]

        if self.group_col is None:
            self.df = pd.get_dummies(self.df.drop(columns=self.info_col), drop_first=True)
            logger.debug('Categorical features turned into dummy')
        else:
            group_col = self.df[[self.group_col]]
            if self.info_col is not None:
                self.df = pd.get_dummies(self.df.drop(columns=[self.group_col]+self.info_col), drop_first=True)
            else:
                self.df = pd.get_dummies(self.df.drop(columns=self.group_col), drop_first=True)
            self.df = pd.concat([self.df, group_col], axis=1)
            logger.debug('Categorical grouped features turned into dummy')
        if self.info_col is not None:
            self.df = pd.concat([self.df, info_col], axis=1)



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

    def lama_feature_select(self):
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
        df = self.df if self.group_col is None else self.df.drop(columns=self.group_col)
        if self.info_col is not None:
            df = df.drop(columns=self.info_col)
        features = feat_select.perform_selection(df=df)
        if self.group_col is None:
            self.features_importance = features
        else:
            self.features_importance = features.append({'Feature': self.group_col,  'Importance': features.Importance.max()},
                                                      ignore_index=True)
        return self.features_importance.sort_values("Importance", ascending=False)

    def _matching(self):
        """Realize matching

        Take in count presence of groups

        Returns:
            Tuple of matched df and ATE

        """
        self.matcher = FaissMatcher(self.df, self.outcome, self.treatment, info_col=self.info_col,
                                    features=self.features_importance,
                                    group_col=self.group_col)
        logger.info('Applying matching')
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
        logger.info('Applying validation of result')

        self.val_dict = {k: [] for k in [self.outcome]}
        self.pval_dict = dict()

        for i in range(n_sim):
            if refuter in ['random_treatment', 'random_feature']:
                if refuter == 'random_treatment':
                    self.df, orig_treatment, self.validate = random_treatment(self.df, self.treatment)
                elif refuter == 'random_feature':
                    self.df, self.validate = random_feature(self.df)
                    if self.features_importance is not None:
                        self.features_importance.append('random_feature')

                self.matcher = FaissMatcher(self.df, self.outcome, self.treatment, info_col=self.info_col,
                                            features=self.features_importance,
                                            group_col=self.group_col, validation=self.validate)
            elif refuter == "subset_refuter":
                df, self.validate = subset_refuter(self.df, self.treatment, fraction)
                self.matcher = FaissMatcher(df, self.outcome, self.treatment, info_col=self.info_col,
                                            features=self.features_importance,
                                            group_col=self.group_col, validation=self.validate)
            else:
                logger.info('Incorrect refuter name!')
                raise NameError('Incorrect refuter name!')

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
        if refuter == 'random_treatment':
            self.df[self.treatment] = orig_treatment
        elif refuter == "random_feature":
            self.df = self.df.drop(columns='random_feature')
            if self.features_importance is not None:
                self.features_importance.pop('random_feature')

        return self.pval_dict

    def estimate(self, features=None):
        """Applies filters and outliers, then matches

        Args:
            features List or feature_importance

        Returns:
            Tuple of matched df and ATE

        """
        if features is not None:
            self.features_importance = features
        return self._matching()
