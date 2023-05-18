import datetime as dt

import faiss
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..utils.metrics import *
from ..utils.psi_pandas import *

POSTFIX = "_matched"
POSTFIX_BIAS = "_matched_bias"

logger = logging.getLogger('Faiss matcher')
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format='[%(asctime)s | %(name)s | %(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    level=logging.INFO
)


class FaissMatcher:
    def __init__(self,
                 df,
                 outcomes,
                 treatment,
                 info_col,
                 features=None,
                 group_col=False,
                 sigma=1.96,
                 validation=None):
        self.df = df
        self.info_col = info_col
        self.columns_del = [outcomes]
        if self.info_col is not None:
            self.columns_del = self.columns_del + [x for x in self.info_col if x in self.df.columns]
        self.outcomes = outcomes
        self.treatment = treatment
        if features is None:
            self.columns_match = list(
                set([x for x in list(self.df.columns) if x not in self.info_col] + [self.treatment, self.outcomes]))
        else:
            try:
                self.columns_match = features['Feature'].tolist() + [self.treatment, self.outcomes]
            except TypeError:
                self.columns_match = features + [self.treatment, self.outcomes]

        self.features_quality = self.df.drop(columns=[self.treatment, self.outcomes] + self.info_col).select_dtypes(
            include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        self.dict_outcome_untreated = {}
        self.dict_outcome_treated = {}
        self.group_col = group_col
        self.treated_index = None
        self.untreated_index = None
        self.orig_treated_index = None
        self.orig_untreated_index = None
        self.results = {}
        self.ATE = None
        self.df_matched = None
        self.sigma = sigma
        self.quality_dict = {}
        self.rep_dict = None
        self.validation = validation

    def _get_split_scalar_data(self, df):
        """Creates split data by treatment column

        Separate treatment column with 1 (treated) an 0 (untreated),
        scales and transforms treatment column

        Args:
            df: pd.DataFrame

        Returns:
            Tuple of dfs treated, untreated; scaled std_treated and std_untreated

        """
        logger.debug('Creating split data by treatment column')

        std_scaler = StandardScaler().fit(df.drop([self.outcomes, self.treatment], axis=1))

        treated = df[df[self.treatment] == 1].drop([self.outcomes, self.treatment], axis=1)
        untreated = df[df[self.treatment] == 0].drop([self.outcomes, self.treatment], axis=1)

        std_treated = pd.DataFrame(std_scaler.transform(treated))
        std_untreated = pd.DataFrame(std_scaler.transform(untreated))

        return treated, untreated, std_treated, std_untreated

    def _predict_outcome(self, std_treated, std_untreated):
        """Func to predict target

        Applies LinearRegression to input arrays,
        calculate biases of treated and untreated values,
        creates dict of y - regular, matched and without bias

        Args:
            std_treated: pd.DataFrame
            std_untreated: pd.DataFrame

        """
        logger.debug('Predicting target by Linear Regression')

        start_time = dt.datetime.now()
        logger.debug('start --')

        self.dict_outcome_untreated = {}
        self.dict_outcome_treated = {}
        df = self.df.drop(columns=self.info_col)

        for outcome in [self.outcomes]:
            if self.group_col is None:
                y_untreated = df[df[self.treatment] == 0][outcome]
                y_treated = df[df[self.treatment] == 1][outcome]
                x_treated = std_treated
                x_untreated = std_untreated
                y_match_untreated = y_untreated.iloc[self.treated_index.ravel()]
                y_match_treated = y_treated.iloc[self.untreated_index.ravel()]
                x_match_treated = x_untreated.iloc[self.treated_index.ravel()]
                x_match_untreated = x_treated.iloc[self.untreated_index.ravel()]
            else:
                y_untreated = df.loc[self.orig_untreated_index.ravel()][outcome]
                y_treated = df.loc[self.orig_treated_index.ravel()][outcome]
                x_treated = df.loc[self.orig_treated_index.ravel()].drop(
                    columns=[self.treatment, outcome, self.group_col])
                x_untreated = df.loc[self.orig_untreated_index.ravel()].drop(
                    columns=[self.treatment, outcome, self.group_col])
                y_match_treated = df.loc[self.untreated_index.ravel()][outcome]
                y_match_untreated = df.loc[self.treated_index.ravel()][outcome]
                x_match_treated = df.loc[self.treated_index.ravel()].drop(
                    columns=[self.treatment, outcome, self.group_col])
                x_match_untreated = df.loc[self.untreated_index.ravel()].drop(
                    columns=[self.treatment, outcome, self.group_col])

            ols0 = LinearRegression().fit(x_untreated, y_untreated)
            ols1 = LinearRegression().fit(x_treated, y_treated)

            bias0 = ols0.predict(x_treated) - ols0.predict(x_match_treated)
            y_match_untreated_bias = y_match_untreated - bias0

            bias1 = ols1.predict(x_untreated) - ols1.predict(x_match_untreated)
            y_match_treated_bias = y_match_treated - bias1

            self.dict_outcome_untreated[outcome] = y_untreated.values
            self.dict_outcome_untreated[outcome + POSTFIX] = y_match_treated.values
            self.dict_outcome_untreated[outcome + POSTFIX_BIAS] = y_match_treated_bias.values

            self.dict_outcome_treated[outcome] = y_treated.values
            self.dict_outcome_treated[outcome + POSTFIX] = y_match_untreated.values
            self.dict_outcome_treated[outcome + POSTFIX_BIAS] = y_match_untreated_bias.values

        end_time = dt.datetime.now()
        total = dt.datetime.strptime(str(end_time - start_time), '%H:%M:%S.%f').strftime('%H:%M:%S')
        logger.debug(f'end -- [work time{total}]')

        return

    def _create_outcome_matched_df(self, dict_outcome, is_treated: bool) -> pd.DataFrame:
        """Matches treated values with treatment column

        Args:
            dict_outcome: dict
            is_treated: bool

        Returns:
            Df with matched values: pd.DataFrame

        """
        df_pred = pd.DataFrame(dict_outcome)
        df_pred[self.treatment] = int(is_treated)
        df_pred[self.treatment + POSTFIX] = int(not is_treated)
        return df_pred

    def _create_features_matched_df(self, index, is_treated: bool):
        """Creates dataframe with matched values

        Args:
            index: int
            is_treated: bool

        Returns:
            Matched dataframe of features: pd.DataFrame

        """
        df = self.df.drop(columns=[self.outcomes])

        if self.group_col is None:
            x1 = df[df[self.treatment] == int(not is_treated)].iloc[index].reset_index()
            x2 = df[df[self.treatment] == int(is_treated)].reset_index()
        else:
            df = df.sort_values(self.group_col)
            x1 = df.loc[index].reset_index()
            if is_treated:
                x2 = df.loc[self.orig_treated_index].reset_index()
            else:
                x2 = df.loc[self.orig_untreated_index].reset_index()
        x1.columns = [col + POSTFIX for col in x2.columns]

        x = pd.concat([x2, x1], axis=1).drop([self.treatment, self.treatment + POSTFIX], axis=1)
        return x

    def _create_matched_df(self):
        """Creates matched df of features and outcome

        Returns:
            Matched dataframe

        """

        df_pred0 = self._create_outcome_matched_df(self.dict_outcome_untreated, False)
        df_pred1 = self._create_outcome_matched_df(self.dict_outcome_treated, True)

        df_matched = pd.concat([df_pred0, df_pred1])

        x_ = self._create_features_matched_df(self.treated_index.ravel(), True)
        x = self._create_features_matched_df(self.untreated_index.ravel(), False)

        x = pd.concat([x_, x])
        df_matched = pd.concat([x.reset_index(drop=True), df_matched.reset_index(drop=True)], axis=1)
        self.df_matched = df_matched

    def calc_ate(self, df, outcome):
        """Calculate ATE - average treatment effect

        Args:
            df: pd.DataFrame
            outcome: pd.Series or {__iter__}

        Returns:
            ATE: int or numpy array

        """
        logger.debug('Calculating ATE')

        ate = np.mean(
            (2 * df[self.treatment] - 1) * (df[outcome] - df[outcome + POSTFIX_BIAS]))

        return ate

    def calc_atc(self, df, outcome):
        """Calculates Average Treatment Effect for the control group (ATC)

        Effect on control group if it was affected

        Args:
            df: pd.DataFrame
            outcome: pd.Series or {__iter__}

        Returns:
            ATC, scaled counts and variances: tuple of numpy arrays

        """
        logger.debug('Calculating ATC')

        df = df[df[self.treatment] == 0]
        N_c = len(df)
        index_c = list(range(N_c)) if self.group_col is None else self.orig_untreated_index
        ITT_c = df[outcome + POSTFIX_BIAS] - df[outcome]
        scaled_counts_c = scaled_counts(N_c, self.treated_index, index_c)

        vars_c = np.repeat(ITT_c.var(), N_c)  # conservative
        atc = np.mean(ITT_c)

        return atc, scaled_counts_c, vars_c

    def calc_att(self, df, outcome):
        """Calculates Average Treatment Effect for the treated (ATT) from pilot

        Args:
            df: pd.DataFrame
            outcome: pd.Series or {__iter__}

        Returns:
            ATT, scaled counts and variances: tuple of numpy arrays

        """
        logger.debug('Calculating ATT')

        df = df[df[self.treatment] == 1]
        N_t = len(df)
        index_t = list(range(N_t)) if self.group_col is None else self.orig_treated_index
        ITT_t = df[outcome] - df[outcome + POSTFIX_BIAS]
        scaled_counts_t = scaled_counts(N_t, self.untreated_index, index_t)

        vars_t = np.repeat(ITT_t.var(), N_t)  # conservative
        att = np.mean(ITT_t)

        return att, scaled_counts_t, vars_t

    def _calculate_ate_all_target(self, df):
        """Creates dicts of all effects

        Args:
            df: pd.DataFrame

        Returns:
            Tuple of dicts with ATE, ATC and ATT

        """
        logger.debug('Creating dicts of all effects: ATE, ATC, ATT')

        att_dict = {}
        atc_dict = {}
        ate_dict = {}
        for outcome in [self.outcomes]:
            ate = self.calc_ate(df, outcome)
            att, scaled_counts_t, vars_t = self.calc_att(df, outcome)
            atc, scaled_counts_c, vars_c = self.calc_atc(df, outcome)
            att_se = calc_att_se(vars_c, vars_t, scaled_counts_c)
            atc_se = calc_atc_se(vars_c, vars_t, scaled_counts_t)
            ate_se = calc_ate_se(vars_c, vars_t, scaled_counts_c, scaled_counts_t)
            ate_dict[outcome] = [ate, ate_se, pval_calc(ate / ate_se), ate - self.sigma * ate_se,
                                 ate + self.sigma * ate_se]
            atc_dict[outcome] = [atc, atc_se, pval_calc(atc / atc_se), atc - self.sigma * atc_se,
                                 atc + self.sigma * atc_se]
            att_dict[outcome] = [att, att_se, pval_calc(att / att_se), att - self.sigma * att_se,
                                 att + self.sigma * att_se]

        self.ATE, self.ATC, self.ATT = ate_dict, atc_dict, att_dict
        self.val_dict = ate_dict
        return

    def matching_quality(self):
        """Estimated the quality of covariates balance and repeat fraction

        Estimates population stability index, Standardized mean difference
        and Kolmogorov-Smirnov test for numeric values. Returns dict of reports.

        Returns:
            dict of reports

        """
        logger.info(f'Estimating quality of matching')

        psi_columns = self.columns_match
        psi_columns.remove(self.treatment)
        psi_data, ks_data, smd_data = matching_quality(self.df_matched, self.treatment, self.features_quality,
                                                       psi_columns)
        rep_dict = {
            'match_control_to_treat': check_repeats(self.treated_index.ravel()),
            'match_treat_to_control': check_repeats(self.untreated_index.ravel())
        }

        self.quality_dict = {
            'psi': psi_data,
            'ks_test': ks_data,
            'smd': smd_data,
            'repeats': rep_dict
        }

        rep_df = pd.DataFrame.from_dict(rep_dict, orient='index').rename(columns={0: 'value'})
        self.rep_dict = rep_df
        logger.info(f'PSI info: \n {psi_data.head(10)} \nshape:{psi_data.shape}')
        logger.info(f'Kolmogorov-Smirnov test info: \n {ks_data.head(10)} \nshape:{ks_data.shape}')
        logger.info(f'Standardised mean difference info: \n {smd_data.head(10)} \nshape:{smd_data.shape}')
        logger.info(f'Repeats info: \n {rep_df.head(10)}')

        return self.quality_dict

    def group_match(self):
        """Matches dfs if it divided by groups

        Returns:
            Tuple of matched df and ATE

        """
        df = self.df.drop(columns=self.info_col)
        df = df.sort_values(self.group_col)
        groups = sorted(df[self.group_col].unique())
        all_treated_matches = {}
        all_untreated_matches = {}
        all_treated_outcome = {}
        all_untreated_outcome = {}
        for group in groups:
            df_group = df[df[self.group_col] == group]
            treated_index = {}
            untreated_index = {}
            temp = df_group[self.columns_match + [self.group_col]]
            temp = temp.loc[:, (temp != 0).any(axis=0)].drop(columns=self.group_col)
            treated, untreated, std_treated, std_untreated = self._get_split_scalar_data(temp)
            for i, ind in enumerate(temp[temp[self.treatment] == 1].index):
                treated_index.update({i: ind})
            for i, ind in enumerate(temp[temp[self.treatment] == 0].index):
                untreated_index.update({i: ind})

            std_treated_np = _transform_to_np(std_treated)
            std_untreated_np = _transform_to_np(std_untreated)
            matches_c = _get_index(std_treated_np, std_untreated_np)
            matches_t = _get_index(std_untreated_np, std_treated_np)
            matches_c = np.array([list(map(lambda x: treated_index[x], l)) for l in matches_c])
            matches_t = np.array([list(map(lambda x: untreated_index[x], l)) for l in matches_t])
            all_treated_matches.update({group: matches_t})
            all_untreated_matches.update({group: matches_c})
            all_treated_outcome.update({group: list(treated_index.values())})
            all_untreated_outcome.update({group: list(untreated_index.values())})
        matches_c = [item for sublist in [i.tolist() for i in list(all_untreated_matches.values())] for item in sublist]
        matches_t = [item for sublist in [i.tolist() for i in list(all_treated_matches.values())] for item in sublist]
        index_c = [item for sublist in [i for i in list(all_untreated_outcome.values())] for item in sublist]
        index_t = [item for sublist in [i for i in list(all_treated_outcome.values())] for item in sublist]
        self.untreated_index = np.array(matches_c)
        self.treated_index = np.array(matches_t)
        self.orig_treated_index = np.array(index_t)
        self.orig_untreated_index = np.array(index_c)
        df_group = df[self.columns_match]
        treated, untreated, std_treated, std_untreated = self._get_split_scalar_data(df_group)
        self._predict_outcome(treated, untreated)
        self._create_matched_df()
        self._calculate_ate_all_target(self.df_matched)

        if self.validation:
            return self.val_dict

        return self.report_view()

    def match(self):
        """Matches df

        Returns:
            Tuple of matched df and metrics ATE, ATC, ATT

        """
        if self.group_col is not None:
            return self.group_match()

        df = self.df[self.columns_match]
        treated, untreated, std_treated, std_untreated = self._get_split_scalar_data(df)

        std_treated_np = _transform_to_np(std_treated)
        std_untreated_np = _transform_to_np(std_untreated)

        untreated_index = _get_index(std_treated_np, std_untreated_np)
        treated_index = _get_index(std_untreated_np, std_treated_np)

        self.untreated_index = untreated_index
        self.treated_index = treated_index

        self._predict_outcome(treated, untreated)

        self._create_matched_df()
        self._calculate_ate_all_target(self.df_matched)

        if self.validation:
            return self.val_dict

        return self.report_view()

    def report_view(self):
        result = (self.ATE, self.ATC, self.ATT)
        self.results = pd.DataFrame([list(x.values())[0] for x in result],
                                    columns=['effect_size', 'std_err', 'p-val', 'ci_lower', 'ci_upper'],
                                    index=['ATE', 'ATC', 'ATT'])
        return self.results


def _get_index(base, new):
    """Getting array of indexes that match new array

    Creating indexes, add them to base array, search them in new array

    Args:
        base: {shape}
        new: array or Any

    Returns:
        Array of indexes

    """
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    print("Finding index")
    indexes = index.search(new, 1)[1]
    return indexes


def _transform_to_np(df):
    """Transforms df to numpy applying PCA analysis

    Args:
        df: pd. DataFrame

    Returns:
        Downsized data: array

    """
    x = df.to_numpy().copy(order='C').astype("float32")
    whiten = faiss.PCAMatrix(x.shape[1], x.shape[1])
    whiten.train(x)
    faiss.vector_to_array(whiten.eigenvalues)
    xt = whiten.apply_py(x)
    return xt


def calc_atx_var(vars_c, vars_t, weights_c, weights_t):
    """Calculates Average Treatment Effect for the treated (ATT) variance

    Args:
        vars_c: {__len__}
        vars_t: {__len__}
        weights_c: {__pow__}
        weights_t: {__pow__}

    Returns:
        ATE variance: int

    """
    N_c, N_t = len(vars_c), len(vars_t)
    summands_c = weights_c ** 2 * vars_c
    summands_t = weights_t ** 2 * vars_t

    return summands_t.sum() / N_t ** 2 + summands_c.sum() / N_c ** 2


def calc_atc_se(vars_c, vars_t, scaled_counts_t):
    """Calculates Average Treatment Effect for the control group (ATC) standard error

    Args:
        vars_c: {__len__}
        vars_t: {__len__}
        scaled_counts_t: Any

    Returns:
        ATC standard error

    """
    N_c, N_t = len(vars_c), len(vars_t)
    weights_c = np.ones(N_c)
    weights_t = (N_t / N_c) * scaled_counts_t

    var = calc_atx_var(vars_c, vars_t, weights_c, weights_t)

    return np.sqrt(var)


def calc_att_se(vars_c, vars_t, scaled_counts_c):
    """Calculates Average Treatment Effect for the treated (ATT) standard error

    Args:
        vars_c: {__len__}
        vars_t: {__len__}
        scaled_counts_c: Any

    Returns:
        ATT standard error

    """
    N_c, N_t = len(vars_c), len(vars_t)
    weights_c = (N_c / N_t) * scaled_counts_c
    weights_t = np.ones(N_t)

    var = calc_atx_var(vars_c, vars_t, weights_c, weights_t)

    return np.sqrt(var)


def calc_ate_se(vars_c, vars_t, scaled_counts_c, scaled_counts_t):
    """Calculates Average Treatment Effect (ATE) standard error

    Args:
        vars_c: {__len__}
        vars_t: {__len__}
        scaled_counts_c: Any
        scaled_counts_t: Any

    Returns:
        ATE standard error

    """
    N_c, N_t = len(vars_c), len(vars_t)
    N = N_c + N_t
    weights_c = (N_c / N) * (1 + scaled_counts_c)
    weights_t = (N_t / N) * (1 + scaled_counts_t)

    var = calc_atx_var(vars_c, vars_t, weights_c, weights_t)

    return np.sqrt(var)


def pval_calc(z):
    """Calculates p-value of norm cdf based on z

    Args:
        z: float

    Returns:
        P-value

    """
    return round(2 * (1 - norm.cdf(abs(z))), 2)


def scaled_counts(N, matches, index):
    """Counts the number of times each subject has appeared as a match

        In the case of multiple matches, each subject only gets partial credit.

        Args:
            N: int or Any
            matches: Series or {__iter__}
            index: list or Any

        Returns:
            Number of times each subject has appeared as a match: int

        """
    s_counts = np.zeros(N)
    index_dict = dict(zip(index, list(range(N))))
    for matches_i in matches:
        scale = 1 / len(matches_i)
        for match in matches_i:
            s_counts[index_dict[match]] += scale
    logger.info(f'Calculated the number of times each subject has appeared as a match: {s_counts}')
    return s_counts
