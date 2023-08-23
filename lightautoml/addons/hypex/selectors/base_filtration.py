"""Function for base filtration of data"""
import pandas as pd
import numpy as np


def const_filtration(X: pd.DataFrame, threshold: float = 0.95) -> list:
    """Function removes features consist of constant value on 95%.

        Args:
            X: related dataset
            threshold: constant fill rate, default is 0.95

        Returns:
            List of filtered columns
    """
    is_const = pd.Series(0, index=X.columns, dtype=np.dtype(bool))
    for col in X.columns:
        # NaNs are not counted using unique (since np.nan != np.nan). Fill them with a unique value:
        cur_col = X.loc[:, col]
        cur_col.loc[~np.isfinite(cur_col)] = cur_col.max() + 1
        # Get values' frequency:
        freqs = cur_col.value_counts(normalize=True)
        is_const[col] = np.any(freqs > threshold)

    selected_features = ~is_const
    if np.sum(selected_features) == 0:
        raise AssertionError("All features were removed by constant filtration.")
    else:
        return X.loc[:, selected_features].columns.to_list()


def nan_filtration(X: pd.DataFrame, threshold: float = 0.8):
    """Function removes features consist of NaN value on 80%.

            Args:
                X: related dataset
                threshold: constant fill rate, default is 0.95

            Returns:
                List of filtered columns
    """
    nan_freqs = np.mean(pd.isnull(X), axis=0)
    is_sparse = nan_freqs > threshold
    selected_features = ~is_sparse
    if np.sum(selected_features) == 0:
        raise AssertionError("All features were removed by nan filtration.")
    else:
        return X.loc[:, selected_features].columns.to_list()
