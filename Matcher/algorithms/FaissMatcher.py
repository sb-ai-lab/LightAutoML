import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import faiss

POSTFIX = "_matched"
POSTFIX_BIAS = "_matched_bias"


class FaissMatcher:
    def __init__(self,
                 df,
                 data,
                 outcomes,
                 treatment,
                 features=None):
        self.df = df
        self.data = data
        self.outcomes = outcomes
        self.treatment = treatment
        if features is None:
            self.feature_list = list(self.df.columns)
            self.feature_list.remove(self.treatment)
            self.feature_list.remove(self.outcomes)
        else:
            self.feature_list = features['Feature'].tolist()
        self.dict_outcome_untreated = {}
        self.dict_outcome_treated = {}
        self.treated_index = None
        self.untreated_index = None
        self.results = {}
        self.ATE = None
        self.n_features = None
        self.df_matched = None

    def _get_split_scalar_data(self, df):
        std_scaler = StandardScaler().fit(df.drop([self.outcomes, self.treatment], 1))

        treated = df[df[self.treatment] == 1].drop([self.outcomes, self.treatment], 1)
        untreated = df[df[self.treatment] == 0].drop([self.outcomes, self.treatment], 1)

        std_treated = pd.DataFrame(std_scaler.transform(treated))
        std_untreated = pd.DataFrame(std_scaler.transform(untreated))

        return treated, untreated, std_treated, std_untreated

    def _transform_to_np(self, df):
        x = df.to_numpy().copy(order='C').astype("float32")
        whiten = faiss.PCAMatrix(x.shape[1], x.shape[1])
        whiten.train(x)
        faiss.vector_to_array(whiten.eigenvalues)
        xt = whiten.apply_py(x)
        return xt

    def _get_index(self, base, new):
        print("Creating index")
        index = faiss.IndexFlatL2(base.shape[1])
        print("Adding index")
        index.add(base)
        print("Finding index")
        indexes = index.search(new, 1)[1].ravel()
        return indexes

    def _predict_outcome(self, std_treated, std_untreated):
        self.dict_outcome_untreated = {}
        self.dict_outcome_treated = {}
        for outcome in [self.outcomes]:
            y_untreated = self.df[self.df[self.treatment] == 0][outcome]
            y_treated = self.df[self.df[self.treatment] == 1][outcome]

            y_match_untreated = y_untreated.iloc[self.treated_index]
            y_match_treated = y_treated.iloc[self.untreated_index]

            ols0 = LinearRegression().fit(std_untreated, y_untreated)
            ols1 = LinearRegression().fit(std_treated, y_treated)

            bias0 = ols0.predict(std_treated) - ols0.predict(std_untreated.iloc[self.treated_index])
            y_match_untreated_bias = y_match_untreated - bias0

            bias1 = ols1.predict(std_untreated) - ols1.predict(std_treated.iloc[self.untreated_index])
            y_match_treated_bias = y_match_treated - bias1

            self.dict_outcome_untreated[outcome] = y_untreated.values
            self.dict_outcome_untreated[outcome + POSTFIX] = y_match_treated.values
            self.dict_outcome_untreated[outcome + POSTFIX_BIAS] = y_match_treated_bias.values

            self.dict_outcome_treated[outcome] = y_treated.values
            self.dict_outcome_treated[outcome + POSTFIX] = y_match_untreated.values
            self.dict_outcome_treated[outcome + POSTFIX_BIAS] = y_match_untreated_bias.values

        return

    def _create_outcome_matched_df(self, dict_outcome, is_treated: bool):
        df_pred = pd.DataFrame(dict_outcome)
        df_pred[self.treatment] = int(is_treated)
        df_pred[self.treatment + POSTFIX] = int(not is_treated)
        return df_pred

    def _create_features_matched_df(self, index, is_treated: bool):
        x1 = self.data[self.data[self.treatment] == int(not is_treated)].iloc[index].reset_index()
        x2 = self.data[self.data[self.treatment] == int(is_treated)].reset_index()
        x1.columns = [col + POSTFIX for col in x2.columns]

        x = pd.concat([x2, x1], 1).drop([self.treatment, self.treatment + POSTFIX], axis=1)
        return x

    def _create_matched_df(self):

        df_pred0 = self._create_outcome_matched_df(self.dict_outcome_untreated, False)
        df_pred1 = self._create_outcome_matched_df(self.dict_outcome_treated, True)

        df_matched = pd.concat([df_pred0, df_pred1])

        x_ = self._create_features_matched_df(self.treated_index, True)
        x = self._create_features_matched_df(self.untreated_index, False)

        x = pd.concat([x_, x])

        df_matched = pd.concat([x.reset_index(drop=True), df_matched.reset_index(drop=True)], 1)
        return df_matched

    def calc_ate(self, df, outcome):
        ate = np.mean(
            (2 * df[self.treatment] - 1) * (df[outcome] - df[outcome + POSTFIX_BIAS]))
        return ate

    def _calculate_ate_all_target(self, df):
        ate_dict = {}
        for outcome in [self.outcomes]:
            ate = self.calc_ate(df, outcome)
            ate_dict[outcome] = ate
        return ate_dict


    def _check_best(self, df_matched, n_features):
        ate_dict = self._calculate_ate_all_target(df_matched)
        if self.n_features is None:
            self.n_features = n_features
            self.ATE = ate_dict
            self.df_matched = df_matched
            return

        diffkeys = sum([1 if ate_dict[k] > self.ATE[k] else -1 for k in ate_dict])

        if diffkeys > 0:
            self.n_features = n_features
            self.ATE = ate_dict
            self.df_matched = df_matched

        if diffkeys == 0:
            if np.array(list(ate_dict.values())).mean() > np.array(list(self.ATE.values())).mean():
                self.n_features = n_features
                self.ATE = ate_dict
                self.df_matched = df_matched

    def match(self):
        for i in range(4, 10):
            df = self.df[self.feature_list[:i] + [self.treatment] + [self.outcomes]]
            treated, untreated, std_treated, std_untreated = self._get_split_scalar_data(df)

            std_treated_np = self._transform_to_np(std_treated)
            std_untreated_np = self._transform_to_np(std_untreated)

            untreated_index = self._get_index(std_treated_np, std_untreated_np)
            treated_index = self._get_index(std_untreated_np, std_treated_np)

            self.untreated_index = untreated_index
            self.treated_index = treated_index

            self._predict_outcome(std_treated, std_untreated)

            df_matched = self._create_matched_df()
            self._check_best(df_matched, i)



        return self.df_matched, self.ATE

