import numpy as np
import scipy.stats as st


def random_treatment(df, treatment):
    """Replaces real treatment with a random placebo treatment.
    Args:
        df: pd.DataFrame
            The initial dataframe
        treatment: str
            The columns name representing the treatment

    Return:
        pd.DataFrame: The modified dataframe with the original treatment replaced
        pd.Series: The original treatment series
        int: A validation flag
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
    """Adds a random feature to the initial dataset.
     Args:
         df: pd.DataFrame
            The initial dataframe

     Return:
         pd.DataFrame: The modified dataframe with an additional random feature
         int: A validation flag
    """

    feature = np.random.normal(0, 1, size=len(df))
    validate = 1
    df["random_feature"] = feature
    return df, validate


def subset_refuter(df, treatment, fraction=0.8):
    """Returns a subset of data with given fraction (default 0.8).

     Args:
         df: pd.DataFrame
            The initial dataframe
         treatment: str
            The column name representing the treatment

     Return:
         pd.DataFrame: The subset of the dataframe
         int: A validation flag
    """

    df = df.groupby(treatment, group_keys=False).apply(lambda x: x.sample(frac=fraction))
    validate = 1
    return df, validate


def test_significance(estimate, simulations) -> float:
    """Performs a significance test for a normal distribution.
    Args:
         estimate: float
            The estimated effect
         simulations: list
            A list of estimated effects from each simulation

     Return:
         float: The p-value of the test
    """

    mean_refute_value = np.mean(simulations)
    std_dev_refute_values = np.std(simulations)
    z_score = (estimate - mean_refute_value) / std_dev_refute_values

    if z_score > 0:  # Right Tail
        p_value = 1 - st.norm.cdf(z_score)
    else:  # Left Tail
        p_value = st.norm.cdf(z_score)

    return p_value
