import logging

import pandas as pd

from ....automl.presets.tabular_presets import TabularAutoML
from ....report import ReportDeco
from ....tasks import Task

logger = logging.getLogger("lama_feature_selector")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
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
            verbose,  # не используется
            generate_report,
            report_dir,
            use_algos,
    ):
        """Initialize the LamaFeatureSelector.

        Args:
            outcome: str
                The target column
            outcome_type: str
                The type of target column
            treatment: str
                The column that determines control and test groups
            timeout: int
                Time limit for the execution of the code
            n_threads: int
                Maximum number of threads to be used
            n_folds: int
                Number of folds for cross-validation
            verbose: bool
                Flag to control the verbosity of the process stages
            generate_report: bool
                Flag to control whether to create a report or not
            report_dir: str
                Directory for storing report files
            use_algos: list[str]
                List of names of LAMA algorithms for feature selection
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
        """Trains a model and returns feature scores.

        This method defines metrics, applies the model, creates a report, and returns feature scores

        Args:
            df: pd.DataFrame
                Input data

        Returns:
            pd.DataFrame: A DataFrame containing the feature scores from the model

        """
        roles = {
            "target": self.outcome,
            "drop": [self.treatment],
        }

        if self.outcome_type == "numeric":
            task_name = "reg"
            loss = "mse"
            metric = "mse"
        elif self.outcome_type == "binary":
            task_name = "binary"
            loss = "logloss"
            metric = "logloss"
        else:
            task_name = "multiclass"
            loss = "crossentropy"
            metric = "crossentropy"

        task = Task(name=task_name, loss=loss, metric=metric)

        automl = TabularAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.n_threads,
            general_params={"use_algos": [self.use_algos]},
            reader_params={"n_jobs": self.n_threads, "cv": self.n_folds, },
        )

        if self.generate_report:
            report = ReportDeco(output_path=self.report_dir)
            automl = report(automl)

        _ = automl.fit_predict(df, roles=roles)

        return automl.model.get_feature_scores()
