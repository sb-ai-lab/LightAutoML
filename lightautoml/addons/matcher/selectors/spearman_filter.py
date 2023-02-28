from scipy.stats import spearmanr

PVALUE = .05


class SpearmanFilter:
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

        selected = []
        columns = df.drop([self.treatment, self.outcome], 1).columns
        for column in columns:
            result = spearmanr(
                df[self.outcome].values,
                df[column].values
            )
            if (abs(result[0] < self.threshold)) & (result[1] < PVALUE):
                selected.append(column)

        columns = selected + [self.treatment, self.outcome]
        df = df[columns]

        return df
