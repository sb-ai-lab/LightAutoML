import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from .faiss_matcher import conditional_covariance


class no_replacement_match():

    def __init__(self, X: pd.DataFrame, a: pd.Series, weights: dict = None):

        self.treatment = a
        self.X = X
        self.weights = weights

    def match(self):

        matches = {}
        cov = conditional_covariance(self.X[self.treatment == 1].values, self.X[self.treatment == 0].values)
        distance_matrix = self._get_distance_matrix(self.X[self.treatment == 1], self.X[self.treatment == 0], cov)
        source_array, neighbor_array_indices, distances = optimally_match_distance_matrix(distance_matrix)
        source_df = self.X[self.treatment == 1].iloc[np.array(source_array)]
        target_df = self.X[self.treatment == 0].iloc[np.array(neighbor_array_indices)]

        matches[1] = self.create_match_df(
            self.treatment, source_df, target_df, distances
        )
        matches[0] = self.create_match_df(
            self.treatment, target_df, source_df, distances
        )

        match_df = pd.concat(matches, sort=True)
        return match_df
        # return np.concatenate(match_df.loc[0].iloc[self.X[self.treatment == 1].index].matches.values)

    def create_match_df(self, base_series, source_df, target_df, distances):
        match_sub_df = pd.DataFrame(
            index=base_series.index,
            columns=[
                "matches",
                "distances",
            ],
            data=base_series.apply(lambda x: pd.Series([[], []])).values,
            dtype="object",
        )

        # matching from source to target: read distances
        match_sub_df.loc[source_df.index] = pd.DataFrame(
            data=dict(
                matches=[[tidx] for tidx in target_df.index],
                distances=distances,
            ),
            index=source_df.index,
        )

        # matching from target to target: fill with zeros
        match_sub_df.loc[target_df.index] = pd.DataFrame(
            data=dict(
                matches=[[tidx] for tidx in target_df.index],
                distances=[[0]] * len(distances),
            ),
            index=target_df.index,
        )
        return match_sub_df

    def _get_metric_dict(self, cov, VI_in_metric_params=True):

        metric_dict = dict(metric='mahalanobis')
        mahalanobis_transform = np.linalg.inv(cov)
        if self.weights is not None:
            features = self.X.columns
            w_list = np.array([self.weights[col] if col in self.weights.keys() else 1 for col in features])
            w_matrix = np.sqrt(np.diag(w_list / w_list.sum()))
            mahalanobis_transform = np.dot(w_matrix, mahalanobis_transform)
        if VI_in_metric_params:
            metric_dict["metric_params"] = {"VI": mahalanobis_transform}
        else:
            metric_dict["VI"] = mahalanobis_transform

        return metric_dict

    def _get_distance_matrix(self, source_df, target_df, cov):
        """
        Create distance matrix for no replacement match.

        Combines metric, caliper and source/target data into a
        precalculated distance matrix which can be passed to
        scipy.optimize.linear_sum_assignment.
        """
        cdist_args = dict(XA=_ensure_array_columnlike(source_df.values),
                          XB=_ensure_array_columnlike(target_df.values))
        cdist_args.update(self._get_metric_dict(cov, False))
        distance_matrix = distance.cdist(**cdist_args)

        return distance_matrix


def optimally_match_distance_matrix(distance_matrix):
    source_array, neighbor_array_indices = linear_sum_assignment(
        distance_matrix
    )
    distances = [
        [distance_matrix[s_idx, t_idx]]
        for s_idx, t_idx in zip(source_array, neighbor_array_indices)
    ]
    return source_array, neighbor_array_indices, distances


def _ensure_array_columnlike(target_array):
    if len(target_array.shape) < 2 or target_array.shape[1] == 1:
        target_array = target_array.reshape(-1, 1)
    return target_array
