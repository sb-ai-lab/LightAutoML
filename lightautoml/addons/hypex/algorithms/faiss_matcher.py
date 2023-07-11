import datetime as dt
from typing import Dict, Union
import faiss
from scipy.stats import norm
from tqdm.auto import tqdm

from ..utils.metrics import *
from ..utils.psi_pandas import *

faiss.cvar.distance_compute_blas_threshold = 100000
POSTFIX = "_matched"
POSTFIX_BIAS = "_matched_bias"

logger = logging.getLogger("Faiss hypex")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class FaissMatcher:
    def __init__(self, df, outcomes, treatment, info_col, features=None, group_col=False, sigma=1.96, validation=None,
                 n_neighbors=10, silent=True, pbar=True):
        """

        Args:
            df - input data: pd.DataFrame
            outcomes - target column/data name: str
            treatment - column/data name with treatment: str
            info_col - informational column name: str
            features - data with name of features
            group_col - column for grouping: str
            sigma - significant level for confidence interval calculation
            validation - flag for validation of estimated ATE with default method 'random_feature'
        """
        self.n_neighbors = n_neighbors
        if group_col is None:
            self.df = df
        else:
            self.df = df.sort_values([treatment, group_col]).reset_index(drop=True)
        self.columns_del = [outcomes]
        if info_col:
            self.info_col = info_col
        else:
            self.info_col = []

        if self.info_col is not None:
            self.columns_del = self.columns_del + [x for x in self.info_col if x in self.df.columns]
        self.outcomes = outcomes
        self.treatment = treatment

        if features is None:
            self.columns_match = list(
                set([x for x in list(self.df.columns) if x not in self.info_col] + [self.treatment, self.outcomes])
            )
        else:
            try:
                self.columns_match = features["Feature"].tolist() + [self.treatment, self.outcomes]
            except TypeError:
                self.columns_match = features + [self.treatment, self.outcomes]

        self.features_quality = (
            self.df.drop(columns=[self.treatment, self.outcomes] + self.info_col)
            .select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"])
            .columns
        )
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
        self.silent = silent
        self.pbar = pbar
        self.tqdm = None

    def _get_split(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        """Creates split data by treatment column

        Separate treatment column with 1 (treated) an 0 (untreated),
        scales and transforms treatment column

        Args:
            df: pd.DataFrame

        Returns:
            Tuple of dfs treated, untreated; scaled std_treated and std_untreated: tuple

        """
        logger.debug("Creating split data by treatment column")

        treated = df[df[self.treatment] == 1].drop([self.outcomes, self.treatment], axis=1)
        untreated = df[df[self.treatment] == 0].drop([self.outcomes, self.treatment], axis=1)

        return treated, untreated

    def _predict_outcome(self, std_treated: pd.DataFrame, std_untreated: pd.DataFrame):
        """Func to predict target

        Applies LinearRegression to input arrays,
        calculate biases of treated and untreated values,
        creates dict of y - regular, matched and without bias

        Args:
            std_treated: pd.DataFrame
            std_untreated: pd.DataFrame

        """
        logger.debug("Predicting target by Linear Regression")

        start_time = dt.datetime.now()
        logger.debug("start --")

        self.dict_outcome_untreated = {}
        self.dict_outcome_treated = {}
        df = self.df.drop(columns=self.info_col)

        for outcome in [self.outcomes]:
            y_untreated = df[df[self.treatment] == 0][outcome].to_numpy()
            y_treated = df[df[self.treatment] == 1][outcome].to_numpy()

            x_treated = std_treated.to_numpy()
            x_untreated = std_untreated.to_numpy()
            y_match_treated = np.array([y_untreated[idx].mean() for idx in self.treated_index])
            y_match_untreated = np.array([y_treated[idx].mean() for idx in self.untreated_index])
            x_match_treated = np.array([x_untreated[idx].mean(0) for idx in self.treated_index])
            x_match_untreated = np.array([x_treated[idx].mean(0) for idx in self.untreated_index])
            bias_coefs_c = bias_coefs(self.untreated_index, y_treated, x_treated)
            bias_coefs_t = bias_coefs(self.treated_index, y_untreated, x_untreated)
            bias_c = bias(x_untreated, x_match_untreated, bias_coefs_c)
            bias_t = bias(x_treated, x_match_treated, bias_coefs_t)

            y_match_treated_bias = y_treated - y_match_treated + bias_t
            y_match_untreated_bias = y_match_untreated - y_untreated - bias_c

            self.dict_outcome_untreated[outcome] = y_untreated
            self.dict_outcome_untreated[outcome + POSTFIX] = y_match_untreated
            self.dict_outcome_untreated[outcome + POSTFIX_BIAS] = y_match_untreated_bias

            self.dict_outcome_treated[outcome] = y_treated
            self.dict_outcome_treated[outcome + POSTFIX] = y_match_treated
            self.dict_outcome_treated[outcome + POSTFIX_BIAS] = y_match_treated_bias

        end_time = dt.datetime.now()
        total = dt.datetime.strptime(str(end_time - start_time), "%H:%M:%S.%f").strftime("%H:%M:%S")
        logger.debug(f"end -- [work time{total}]")

    def _create_outcome_matched_df(self, dict_outcome: dict, is_treated: bool) -> pd.DataFrame:
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

    def _create_features_matched_df(self, index: np.array, is_treated: bool) -> pd.DataFrame:
        """Creates dataframe with matched values

        Args:
            index: int
            is_treated: bool

        Returns:
            Matched dataframe of features: pd.DataFrame

        """
        df = self.df.drop(columns=[self.outcomes]+self.info_col)

        if self.group_col is None:
            filtered = df.loc[df[self.treatment] == int(not is_treated)].values
            untreated_df = pd.DataFrame(data=np.array([filtered[idx].mean(axis=0) for idx in index]),
                                        columns=df.columns)
            untreated_df['index'] = pd.Series(list(index))
            treated_df = df[df[self.treatment] == int(is_treated)].reset_index()
        else:
            filtered = df.loc[df[self.treatment] == int(not is_treated)]
            cols_untreated = [col for col in filtered.columns if col != self.group_col]
            filtered = filtered.drop(columns=self.group_col).to_numpy()
            untreated_df = pd.DataFrame(data=np.array([filtered[idx].mean(axis=0) for idx in index]),
                                        columns=cols_untreated)
            untreated_df['index'] = pd.Series(list(index))
            treated_df = df[df[self.treatment] == int(is_treated)].reset_index()
            grp = treated_df[self.group_col]
            untreated_df[self.group_col] = grp
        untreated_df.columns = [col + POSTFIX for col in untreated_df.columns]

        x = pd.concat([treated_df, untreated_df], axis=1).drop(
            columns=[self.treatment, self.treatment + POSTFIX], axis=1)
        return x

    def _create_matched_df(self):
        """Creates matched df of features and outcome

        Returns:
            Matched dataframe

        """
        df_pred_treated = self._create_outcome_matched_df(self.dict_outcome_treated, True)
        df_pred_untreated = self._create_outcome_matched_df(self.dict_outcome_untreated, False)

        df_matched = pd.concat([df_pred_treated, df_pred_untreated])

        treated_x = self._create_features_matched_df(self.treated_index, True)
        untreated_x = self._create_features_matched_df(self.untreated_index, False)

        untreated_x = pd.concat([treated_x, untreated_x])

        columns = list(untreated_x.columns) + list(df_matched.columns)

        df_matched = pd.concat([untreated_x, df_matched], axis=1, ignore_index=True)
        df_matched.columns = columns

        self.df_matched = df_matched

    def calc_atc(self, df: pd.DataFrame, outcome: str) -> ():
        """Calculates Average Treatment Effect for the control group (ATC)

        Effect on control group if it was affected

        Args:
            df: pd.DataFrame
            outcome: pd.Series or {__iter__}

        Returns:
            ATC, scaled counts and variances: tuple of numpy arrays

        """
        logger.debug("Calculating ATC")

        df = df[df[self.treatment] == 0]
        N_c = len(df)
        ITT_c = df[outcome + POSTFIX_BIAS]
        scaled_counts_c = scaled_counts(N_c, self.treated_index, self.silent)

        vars_c = np.repeat(ITT_c.var(), N_c)  # conservative
        atc = ITT_c.mean()

        return atc, scaled_counts_c, vars_c

    def calc_att(self, df: pd.DataFrame, outcome: str) -> tuple:
        """Calculates Average Treatment Effect for the treated (ATT) from pilot

        Args:
            df: pd.DataFrame
            outcome: str

        Returns:
            ATT, scaled counts and variances: tuple of numpy arrays

        """
        logger.debug("Calculating ATT")

        df = df[df[self.treatment] == 1]
        N_t = len(df)
        ITT_t = df[outcome + POSTFIX_BIAS]
        scaled_counts_t = scaled_counts(N_t, self.untreated_index, self.silent)

        vars_t = np.repeat(ITT_t.var(), N_t)  # conservative
        att = ITT_t.mean()

        return att, scaled_counts_t, vars_t

    def _calculate_ate_all_target(self, df: pd.DataFrame):
        """Creates dicts of all effects

        Args:
            df: pd.DataFrame

        """
        logger.debug("Creating dicts of all effects: ATE, ATC, ATT")

        att_dict = {}
        atc_dict = {}
        ate_dict = {}
        N = len(df)
        N_t = df[self.treatment].sum()
        N_c = N - N_t

        for outcome in [self.outcomes]:
            att, scaled_counts_t, vars_t = self.calc_att(df, outcome)
            atc, scaled_counts_c, vars_c = self.calc_atc(df, outcome)
            ate = (N_c / N) * atc + (N_t / N) * att

            att_se = calc_att_se(vars_c, vars_t, scaled_counts_c)
            atc_se = calc_atc_se(vars_c, vars_t, scaled_counts_t)
            ate_se = calc_ate_se(vars_c, vars_t, scaled_counts_c, scaled_counts_t)

            ate_dict[outcome] = [
                ate,
                ate_se,
                pval_calc(ate / ate_se),
                ate - self.sigma * ate_se,
                ate + self.sigma * ate_se,
            ]
            atc_dict[outcome] = [
                atc,
                atc_se,
                pval_calc(atc / atc_se),
                atc - self.sigma * atc_se,
                atc + self.sigma * atc_se,
            ]
            att_dict[outcome] = [
                att,
                att_se,
                pval_calc(att / att_se),
                att - self.sigma * att_se,
                att + self.sigma * att_se,
            ]

        self.ATE, self.ATC, self.ATT = ate_dict, atc_dict, att_dict
        self.val_dict = ate_dict

    def matching_quality(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Estimated the quality of covariates balance and repeat fraction

        Estimates population stability index, Standardized mean difference
        and Kolmogorov-Smirnov test for numeric values. Returns dict of reports.

        Returns:
            dict of reports

        """
        if self.silent:
            logger.debug(f"Estimating quality of matching")
        else:
            logger.info(f"Estimating quality of matching")

        psi_columns = self.columns_match
        psi_columns.remove(self.treatment)
        psi_data, ks_data, smd_data = matching_quality(
            self.df_matched, self.treatment, sorted(self.features_quality), sorted(psi_columns), self.silent
        )

        rep_dict = {
            "match_control_to_treat": check_repeats(np.concatenate(self.treated_index), silent=self.silent),
            "match_treat_to_control": check_repeats(np.concatenate(self.untreated_index), silent=self.silent),
        }

        self.quality_dict = {"psi": psi_data, "ks_test": ks_data, "smd": smd_data, "repeats": rep_dict}

        rep_df = pd.DataFrame.from_dict(rep_dict, orient="index").rename(columns={0: "value"})
        self.rep_dict = rep_df

        if self.silent:
            logger.debug(f"PSI info: \n {psi_data.head(10)} \nshape:{psi_data.shape}")
            logger.debug(f"Kolmogorov-Smirnov test info: \n {ks_data.head(10)} \nshape:{ks_data.shape}")
            logger.debug(f"Standardised mean difference info: \n {smd_data.head(10)} \nshape:{smd_data.shape}")
            logger.debug(f"Repeats info: \n {rep_df.head(10)}")
        else:
            logger.info(f"PSI info: \n {psi_data.head(10)} \nshape:{psi_data.shape}")
            logger.info(f"Kolmogorov-Smirnov test info: \n {ks_data.head(10)} \nshape:{ks_data.shape}")
            logger.info(f"Standardised mean difference info: \n {smd_data.head(10)} \nshape:{smd_data.shape}")
            logger.info(f"Repeats info: \n {rep_df.head(10)}")


        return self.quality_dict

    def group_match(self):
        """Matches dfs if it divided by groups

        Returns:
            Tuple of matched df and ATE

        """
        df = self.df.drop(columns=self.info_col)
        groups = sorted(df[self.group_col].unique())
        matches_c = []
        matches_t = []
        group_arr_c = df[df[self.treatment] == 0][self.group_col].to_numpy()
        group_arr_t = df[df[self.treatment] == 1][self.group_col].to_numpy()
        treat_arr_c = df[df[self.treatment] == 0][self.treatment].to_numpy()
        treat_arr_t = df[df[self.treatment] == 1][self.treatment].to_numpy()

        if self.pbar:
            self.tqdm = tqdm(total=len(groups)*2)

        for group in groups:
            df_group = df[df[self.group_col] == group]
            temp = df_group[self.columns_match + [self.group_col]]
            temp = temp.loc[:, (temp != 0).any(axis=0)].drop(columns=self.group_col)
            treated, untreated = self._get_split(temp)

            std_treated_np, std_untreated_np = _transform_to_np(treated, untreated)

            if self.pbar:
                self.tqdm.set_description(desc=f"Get untreated index by group {group}")
            matches_u_i = _get_index(std_treated_np, std_untreated_np, self.n_neighbors)

            if self.pbar:
                self.tqdm.update(1)
                self.tqdm.set_description(desc=f"Get treated index by group {group}")
            matches_t_i = _get_index(std_untreated_np, std_treated_np, self.n_neighbors)
            if self.pbar:
                self.tqdm.update(1)
                self.tqdm.refresh()

            group_mask_c = (group_arr_c == group)
            group_mask_t = (group_arr_t == group)
            matches_c_mask = np.arange(treat_arr_t.shape[0])[group_mask_t]
            matches_u_i = [matches_c_mask[i] for i in matches_u_i]
            matches_t_mask = np.arange(treat_arr_c.shape[0])[group_mask_c]
            matches_t_i = [matches_t_mask[i] for i in matches_t_i]
            matches_c.extend(matches_u_i)
            matches_t.extend(matches_t_i)

        if self.pbar:
            self.tqdm.close()

        self.untreated_index = np.array(matches_c)
        self.treated_index = np.array(matches_t)
        df_group = df[self.columns_match].drop(columns=self.group_col)
        treated, untreated = self._get_split(df_group)
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
        treated, untreated = self._get_split(df)

        std_treated_np, std_untreated_np = _transform_to_np(treated, untreated)

        if self.pbar:
            self.tqdm = tqdm(total=len(std_treated_np) + len(std_untreated_np))
            self.tqdm.set_description(desc="Get untreated index")

        untreated_index = _get_index(std_treated_np, std_untreated_np, self.n_neighbors)

        if self.pbar:
            self.tqdm.update(len(std_treated_np))
            self.tqdm.set_description(desc="Get treated index")
        treated_index = _get_index(std_untreated_np, std_treated_np, self.n_neighbors)

        if self.pbar:
            self.tqdm.update(len(std_untreated_np))
            self.tqdm.refresh()
            self.tqdm.close()

        self.untreated_index = untreated_index
        self.treated_index = treated_index

        self._predict_outcome(treated, untreated)

        self._create_matched_df()
        self._calculate_ate_all_target(self.df_matched)

        if self.validation:
            return self.val_dict

        return self.report_view()

    def report_view(self) -> pd.DataFrame:
        result = (self.ATE, self.ATC, self.ATT)
        self.results = pd.DataFrame(
            [list(x.values())[0] for x in result],
            columns=["effect_size", "std_err", "p-val", "ci_lower", "ci_upper"],
            index=["ATE", "ATC", "ATT"],
        )
        return self.results


def _get_index(base, new, n_neighbors: int):
    """Getting array of indexes that match new array

    Creating indexes, add them to base array, search them in new array

    Args:
        base: {shape}
        new: array or Any

    Returns:
        Array of indexes"""

    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    dist, indexes = index.search(new, n_neighbors)
    map_func = lambda x: np.where(x == x[0])[0]
    equal_dist = list(map(map_func, dist))
    f2 = lambda x, y: x[y]
    indexes = np.array([f2(i, j) for i, j in zip(indexes, equal_dist)])
    return indexes


def _transform_to_np(treated, untreated):
    """Transforms df to numpy applying PCA analysis

    Args:
        untreated - control subset: pd. DataFrame
        treated - test subset: pd. DataFrame

    Returns:
        Downsized data: Tuple[pd.DataFrame, pd.DataFrame]

    """
    xc = untreated.to_numpy()
    xt = treated.to_numpy()

    cov_c = np.cov(xc, rowvar=False, ddof=0)
    cov_t = np.cov(xt, rowvar=False, ddof=0)
    cov = (cov_c + cov_t) / 2

    L = np.linalg.cholesky(cov)
    mahalanobis_transform = np.linalg.inv(L)
    yc = np.dot(xc, mahalanobis_transform.T)
    yt = np.dot(xt, mahalanobis_transform.T)

    return yt.copy(order='C').astype("float32"), yc.copy(order='C').astype("float32")


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


def scaled_counts(N: int, matches, silent=True) -> np.array:
    """Counts the number of times each subject has appeared as a match

    In the case of multiple matches, each subject only gets partial credit.

    Args:
        N - length of original treated/control group: int
        matches - matched indexes from control/treated group: Series
        index - indexes from control/treated group: list or Any

    Returns:
        Number of times each subject has appeared as a match: int"""

    s_counts = np.zeros(N)

    for matches_i in matches:
        scale = 1 / len(matches_i)
        for match in matches_i:
            s_counts[match] += scale

    if silent:
        logger.debug(f"Calculated the number of times each subject has appeared as a match: {len(s_counts)}")
    else:
        logger.info(f"Calculated the number of times each subject has appeared as a match: {len(s_counts)}")

    return s_counts


def bias_coefs(matches, Y_m, X_m):
    """Computes OLS coefficient in bias correction regression. Constructs
    data for regression by including (possibly multiple times) every
    observation that has appeared in the matched sample."""

    flat_idx = np.concatenate(matches)
    N, K = len(flat_idx), X_m.shape[1]

    Y = Y_m[flat_idx]
    X = np.empty((N, K + 1))
    X[:, 0] = 1  # intercept term
    X[:, 1:] = X_m[flat_idx]

    return np.linalg.lstsq(X, Y)[0][1:]  # don't need intercept coef


def bias(X, X_m, coefs):
    """Computes bias correction term, which is approximated by the dot
    product of the matching discrepancy (i.e., X-X_matched) and the
    coefficients from the bias correction regression."""

    bias_list = [(X_j - X_i).dot(coefs) for X_i, X_j in zip(X, X_m)]

    return np.array(bias_list)
