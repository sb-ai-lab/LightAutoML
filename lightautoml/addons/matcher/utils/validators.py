import numpy as np
import scipy.stats as st


def random_treatment(df, treatment):
    """Replacing real treatment with random placebo treatment.
    Args:
        df - initial dataframe: pd.DataFrame
        treatment - treatment column name: str

    Return:
        Modified dataframe, original treatment series and validation flag.
    """

    prop1 = df[treatment].sum() / df.shape[0]
    prop0 = 1 - prop1
    new_treatment = np.random.choice([0, 1], size=df.shape[0], p=[prop0, prop1])
    validate = 1
    orig_treatment = df[treatment]
    df = df.drop(columns=treatment)
    df[treatment] = new_treatment
    return df, orig_treatment, validate


def random_feature(df):
    """Function adds random feature to initial dataset.
     Args:
         df - initial dataframe: pd.DataFrame

     Return:
         Modified dataframe, validation flag.
    """

    feature = np.random.normal(0, 1, size=len(df))
    validate = 1
    df['random_feature'] = feature
    return df, validate


def subset_refuter(df, treatment, fraction=0.8):
    """Functions returns subset of data with given fraction (by default 0.8).
     Args:
         df - initial dataframe: pd.DataFrame
         treatment - treatment column name: str

     Return:
         Modified dataframe, validation flag.
    """

    df = df.groupby(treatment, group_keys=False).apply(lambda x: x.sample(frac=fraction))
    validate = 1
    return df, validate


def test_significance(estimate, simulations):
    """Significance test for normal distribution
    Args:
         estimate - estimated effect: float
         simulations - list of estimated effects on each simulation: list

     Return:
         p-value: float
    """

    mean_refute_value = np.mean(simulations)
    std_dev_refute_values = np.std(simulations)
    z_score = (estimate - mean_refute_value) / std_dev_refute_values

    if z_score > 0:  # Right Tail
        p_value = 1 - st.norm.cdf(z_score)
    else:  # Left Tail
        p_value = st.norm.cdf(z_score)

    return p_value
