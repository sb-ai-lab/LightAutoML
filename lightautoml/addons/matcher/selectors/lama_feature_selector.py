import pandas as pd

from ....automl.presets.tabular_presets import TabularAutoML
from ....report import ReportDeco
from ....tasks import Task

import logging

logger = logging.getLogger('lama_feature_selector')
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format='[%(asctime)s | %(name)s | %(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.INFO
)


class LamaFeatureSelector:
    def __init__(
            self,
            outcome,
            outcome_type,
            treatment,
            timeout,
            n_threads,
            n_folds,
            verbose, # не используется
            generate_report,
            report_dir,
            use_algos,
    ):
        """

        Args:
            outcome: target column
            outcome_type: type of target column
            treatment: column determine control and test groups
            timeout: limit work time of code
            n_threads: maximum number of threads
            n_folds: number of folds for cross-validation
            verbose: flag to show process stages
            generate_report: flag to create report
            report_dir: folder for report files
            use_algos: list of names of LAMA algorithms for feature selection
        """
        self.outcome = outcome
        self.outcome_type = outcome_type
        self.treatment = treatment
        self.use_algos = use_algos
        self.timeout = timeout
        self.n_threads = n_threads
        self.n_folds = n_folds
        self.verbose = verbose
        self.generate_report = generate_report
        self.report_dir = report_dir

    def perform_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realize model and returns feature scores

        Defines metrics, applies the model, creates report
        and gives feature scores

        Args:
            df: pd.DataFrame

        Returns:
            feature scores of model: pd.DataFrame

        """
        logger.info('Getting feature scores')
        roles = {
            'target': self.outcome,
            'drop': [self.treatment],
        }

        if self.outcome_type == 'numeric':
            task_name = 'reg'
            loss = 'mse'
            metric = 'mse'
        elif self.outcome_type == 'binary':
            task_name = 'binary'
            loss = 'logloss'
            metric = 'logloss'
        else:
            task_name = 'multiclass'
            loss = 'crossentropy'
            metric = 'crossentropy'

        task = Task(
            name=task_name,
            loss=loss,
            metric=metric
        )

        automl = TabularAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.n_threads,
            general_params={
                'use_algos': [self.use_algos]
            },
            reader_params={
                'n_jobs': self.n_threads,
                'cv': self.n_folds,
            }
        )

        if self.generate_report:
            report = ReportDeco(output_path=self.report_dir)
            automl = report(automl)

        _ = automl.fit_predict(df, roles=roles)

        return automl.model.get_feature_scores()
