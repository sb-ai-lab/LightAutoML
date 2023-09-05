"""Outliers filter."""
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
    """Class of Outliers Filter. It creates a row indices that should be deleted by percentile."""

    def __init__(self, interquartile_coeff, mode_percentile, min_percentile, max_percentile):
        """Initializes the OutliersFilter.

        Args:
            interquartile_coeff:
                Coefficient for the interquartile range to determine outliers
            mode_percentile:
                If True, outliers are determined by custom percentiles
            min_percentile:
                The lower percentile. Values below this percentile are considered outliers.
            max_percentile:
                The upper percentile. Values above this percentile are considered outliers
        """
        self.interquartile_coeff = interquartile_coeff
        self.mode_percentile = mode_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def perform_filter(self, df: pd.DataFrame, interquartile: bool = True) -> set:
        """Identifies rows with outliers.

        This method creates a set of row indices to be removed, which contains values less than
        `min_percentile` and larger than `max_percentile` (if `mode_percentile` is True), or values
        smaller than the 0.2 and larget than 0.8 (if `mode_percentile` is False)

        Args:
            df:
                The input DataFrame
            interquartile:
                If True, uses the interquartile range to determine outliers. Defaults to True

        Returns:
            The set of row indices with outliers
        """
        columns_names = df.select_dtypes(include="number").columns
        rows_for_del = []
        for column in columns_names:
            if self.mode_percentile:
                min_value = df[column].quantile(self.min_percentile)
                max_value = df[column].quantile(self.max_percentile)
            elif interquartile:
                upper_quantile = df[column].quantile(0.8)
                lower_quantile = df[column].quantile(0.2)

                interquartile_range = upper_quantile - lower_quantile
                min_value = lower_quantile - self.interquartile_coeff * interquartile_range
                max_value = upper_quantile + self.interquartile_coeff * interquartile_range
            else:
                mean_value = df[column].mean()
                standard_deviation = df[column].std()
                nstd_lower, nstd_upper = 3, 3

                min_value = mean_value - nstd_lower * standard_deviation
                max_value = mean_value + nstd_upper * standard_deviation

            rows_for_del_column = (df[column] < min_value) | (df[column] > max_value)
            rows_for_del_column = df.index[rows_for_del_column].tolist()
            rows_for_del.extend(rows_for_del_column)
        rows_for_del = set(rows_for_del)
        logger.info(f"Drop {len(rows_for_del)} rows")

        return rows_for_del
