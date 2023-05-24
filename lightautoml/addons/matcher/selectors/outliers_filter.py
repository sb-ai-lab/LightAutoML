import logging

import pandas as pd

logger = logging.getLogger("outliers_filter")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class OutliersFilter:
    def __init__(self, interquartile_coeff, mode_percentile, min_percentile, max_percentile):
        """

        Args:
            interquartile_coeff: interquartile coefficient - percent for drop outliers
            mode_percentile: flag to drop outliers by custom percentiles
            min_percentile: minimum percentile to drop outliers
            max_percentile: maximum percentile to drop outliers
        """
        self.interquartile_coeff = interquartile_coeff
        self.mode_percentile = mode_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def perform_filter(self, df: pd.DataFrame) -> set:
        """Drops outlayers

        Creates set of rows to be deleted,
        that contains values less than min_percentile
        and larger than max_percentile if mode_percentile is true
        or 25 percentile and larger than 75 percentile if not

        Args:
            df: pd.DataFrame

        Returns:
            rows_for_del: set

        """
        columns_names = df.select_dtypes(include="number").columns
        rows_for_del = []
        for column in columns_names:
            if self.mode_percentile:
                min_value = df[column].quantile(self.min_percentile)
                max_value = df[column].quantile(self.max_percentile)
            else:
                high_quantile = df[column].quantile(0.75)
                low_quantile = df[column].quantile(0.25)

                interquartile_range = high_quantile - low_quantile
                min_value = low_quantile - self.interquartile_coeff * interquartile_range
                max_value = high_quantile + self.interquartile_coeff * interquartile_range

            rows_for_del_column = (df[column] < min_value) | (df[column] > max_value)
            rows_for_del_column = df.index[rows_for_del_column].tolist()
            rows_for_del.extend(rows_for_del_column)
        rows_for_del = set(rows_for_del)
        logger.info(f"Drop {len(rows_for_del)} rows")

        return rows_for_del
