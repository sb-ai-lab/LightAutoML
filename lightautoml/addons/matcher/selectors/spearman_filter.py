from scipy.stats import spearmanr
import logging

PVALUE = .05

logger = logging.getLogger('spearman_filter')
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format='[%(asctime)s | %(name)s | %(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.INFO
)


class SpearmanFilter:
    """Class for filter columns by value of Spearman correlation coefficient

    Example:
        filter = SpearmanFilter(
            outcome = df[['outcome']],
            treatment = df[['treatment']],
            threshold = df[['threshold']]
        )

        df = filter.perform_filter(df)

    """

    def __init__(
            self,
            outcome,
            treatment,
            threshold
    ):
        self.outcome = outcome
        self.treatment = treatment
        self.threshold = threshold

    def perform_filter(self, df):
        """Filter columns by correlation with outcome column.

        Correlation tests by Spearman coefficient,
        that should be less than threshold, and p-value=0.5

        Args:
            df: pd.DataFrame

        Returns:
            pd.DataFrame with columns, non-correlated with outcome column

        """
        selected = []
        columns = df.drop([self.treatment, self.outcome], 1).columns
        for column in columns:
            result = spearmanr(
                df[self.outcome].values,
                df[column].values
            )
            if (abs(result[0] < self.threshold)) & (result[1] < PVALUE):
                selected.append(column)

        logger.info(f'Drop columns {list(set(columns) - set(selected))}')
        columns = selected + [self.treatment, self.outcome]
        df = df[columns]

        return df
