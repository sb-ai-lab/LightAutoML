import pandas as pd
from scipy.stats import spearmanr
import logging

PVALUE = 0.05

logger = logging.getLogger("spearman_filter")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class SpearmanFilter:
    """A class to filter columns based on the Spearman correlation coefficient.

    The class is utilized to filter dataframe columns that do not exhibit a significant
    correlation (based on a provided threshold) with a specified outcome column.
    The significance of the correlation is determined using the Spearman correlation coefficient
    and a p-value threshold of 0.05
    """
    def __init__(self, outcome: str, treatment: str, threshold: float):
        """
        Initialize spearman filter.

        Args:
            outcome: str
                The name of target column
            treatment: str
                The name of the column that determines control and test groups
            threshold: float
                The threshold for the Spearman correlation coefficient filter
        """
        self.outcome: str = outcome
        self.treatment: str  = treatment
        self.threshold: float  = threshold

    def perform_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters columns based on their correlation with the outcome column.

        The method tests the correlation using the Spearman correlation coefficient.
        Columns that have an absolute correlation coefficient value less than the provided threshold,
        and a p-value less than 0.05, are considered insignificant and are removed from the dataframe

        Args:
            df: pd.DataFrame
                The input DataFrame

        Returns:
            pd.DataFrame: The filtered DataFrame, containing only columns that
            are significantly correlated with the outcome column
        """
        selected = []
        columns = df.drop([self.treatment, self.outcome], 1).columns
        for column in columns:
            result = spearmanr(df[self.outcome].values, df[column].values)
            if (abs(result[0] < self.threshold)) and (result[1] < PVALUE):
                selected.append(column)

        logger.info(f"Drop columns {list(set(columns) - set(selected))}")

        columns = selected + [self.treatment, self.outcome]
        df = df[columns]

        return df
