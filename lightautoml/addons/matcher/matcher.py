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
            input_data,
            outcome,
            treatment,
            outcome_type='numeric',
            group_col=None,
            info_col=None,
            required_col=None, # похоже не используется, лол 😂 (с)Дима
            generate_report=GENERATE_REPORT,
            report_feat_select_dir=REPORT_FEAT_SELECT_DIR,
            report_prop_score_dir=REPORT_PROP_SCORE_DIR, # похоже не используется, лол 😂 (с)Дима
            report_matcher_dir=REPORT_PROP_MATCHER_DIR, # похоже не используется, лол 😂 (с)Дима
            timeout=TIMEOUT,
            n_threads=N_THREADS,
            n_folds=N_FOLDS,
            verbose=VERBOSE,
            use_algos=None,
            same_target_threshold=SAME_TARGET_THRESHOLD,
            interquartile_coeff=OUT_INTER_COEFF,
            drop_outliers_by_percentile=OUT_MODE_PERCENT,
            min_percentile=OUT_MIN_PERCENT,
            max_percentile=OUT_MAX_PERCENT
    ):
        """

        Args:
            input_data: датафрейм входной
            outcome: колонка с таргетом?
            treatment: колонка с обозначением контрольной и тестовой группы?
            outcome_type: тип колонки с таргетом?
            group_col: колонка, по которой производится группировка
            info_col: ???
            required_col: не используется
            generate_report: флаг создания отчета
            report_feat_select_dir: folder for report files
            report_prop_score_dir: не используется
            report_matcher_dir: не используется
            timeout: ограничение времени выполнения
            n_threads: максимальное количество потоков
            n_folds: количество фолдов для кросс-валидации
            verbose: флаг вывода этапов процесса
            use_algos: название алгоритма в LAMA для отбора фичей
            same_target_threshold: threshold for correlation coefficient filter (Spearman)
            interquartile_coeff: интерквартильный коэффициент
            drop_outliers_by_percentile: флаг удаления выбросов по перцентилям
            min_percentile: минимальный процент, ниже которого удаляются выбросы
            max_percentile: максимальный процент, выше которого удаляются выбросы
        """
        if use_algos is None:
            use_algos = USE_ALGOS
        self.input_data = input_data
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
        self.mode_percentile = drop_outliers_by_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.info_col = info_col
        self.features_importance = None
        self._preprocessing_data()
        self.matcher = None
        self.val_dict = {self.outcome: []}
        self.pval_dict = None
        self.new_treatment = None
        self.validate = None

    def _preprocessing_data(self):
        """Turns categorical features into dummy.
        """
        if self.group_col is None:
            self.input_data = pd.get_dummies(self.input_data, drop_first=True)
            logger.debug('Categorical features turned into dummy')
        else:
            group_col = self.input_data[[self.group_col]]
            self.input_data = pd.get_dummies(
                self.input_data.drop(columns=self.group_col),
                drop_first=True
            )
            self.input_data = pd.concat([self.input_data, group_col], axis=1)
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

        self.input_data = same_filter.perform_filter(self.input_data)

    def _outliers_filter(self):
        """Deletes outliers

        If drop_outliers_by_percentile is true, leaves values between min and max
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

        rows_for_del = out_filter.perform_filter(self.input_data)
        self.input_data = self.input_data.drop(rows_for_del, axis=0)

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
        df = self.input_data if self.group_col is None else self.input_data.drop(columns=self.group_col)

        if self.info_col is not None:
            df = df.drop(columns=self.info_col)

        features = feat_select.perform_selection(df=df)
        self.features_importance = features
        return self.features_importance

    def _matching(self):
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
            group_col=self.group_col
        )
        logger.info('Applying matching')
        self.results = self.matcher.match()

        self.quality_result = self.matcher.matching_quality()

        return self.results, self.quality_result

    def validate_result(self, n_sim=10) -> dict:
        """Validates estimated effect

        Validates estimated effect by replacing real treatment with random
        placebo treatment.
        Estimated effect must be dropped to zero

        Args:
            n_sim - number of simulations: int

        Returns:
            self.pval_dict - dict of p-values: dict

        """
        logger.info('Applying validation of result')

        for i in range(n_sim):
            prop1 = self.input_data[self.treatment].sum() / self.input_data.shape[0]
            prop0 = 1 - prop1
            self.new_treatment = np.random.choice(
                [0, 1],
                size=self.input_data.shape[0],
                p=[prop0, prop1]
            )
            self.validate = 1
            self.input_data = self.input_data.drop(columns=self.treatment)
            self.input_data[self.treatment] = self.new_treatment
            self.matcher = FaissMatcher(
                self.input_data, self.outcome,
                self.treatment,
                info_col=self.info_col,
                features=self.features_importance,
                group_col=self.group_col,
                validation=self.validate
            )

            sim = self.matcher.match()

            for key in self.val_dict.keys():
                self.val_dict[key].append(sim[key][0])

        self.pval_dict = dict()

        for outcome in [self.outcome]:
            self.pval_dict.update({outcome: np.mean(self.val_dict[outcome])})

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
