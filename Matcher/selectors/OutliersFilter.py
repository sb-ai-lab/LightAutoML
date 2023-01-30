

class OutliersFilter:
    def __init__(
            self,
            interquartile_coeff,
            mode_percentile,
            min_percentile,
            max_percentile
    ):
        self.interquartile_coeff = interquartile_coeff
        self.mode_percentile = mode_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def _delete_outliers(self, df):
        columns_names = df.select_dtypes(include='number').columns
        rows_for_del = []
        for column in columns_names:
            if self.mode_percentile:
                min_value = df[column].quantile(self.min_percentile)
                max_value = df[column].quantile(self.max_percentile)
            else:
                interquartile_range = df[column].quantile(.75)-df[column].quantile(.25)
                min_value = df[column].quantile(.25) - self.interquartile_coeff * interquartile_range
                max_value = df[column].quantile(.75) + self.interquartile_coeff * interquartile_range
            rows_for_del_column = (df[column] < min_value) | (df[column] > max_value)
            rows_for_del_column = df.index[rows_for_del_column].tolist()
            rows_for_del.extend(rows_for_del_column)
        rows_for_del = set(rows_for_del)

        return rows_for_del

    def perform_filter(self, df):
        rows_for_del = self._delete_outliers(df)
        df = df.drop(rows_for_del, 0)
        return  df

