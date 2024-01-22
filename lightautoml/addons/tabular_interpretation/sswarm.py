"""SSWARM."""

from typing import Union
from typing import List
from typing import Set

import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from copy import copy
from inspect import getfullargspec


class SSWARM:
    """Fast computation of shapley values.

    Base on the Stratified SWARM algorithm.
    Origin:
        Title: Approximating the Shapley Value without Marginal Contributions.
        Authors: Patrick Kolpaczki and Viktor Bengs and Maximilian Muschalik and Eyke Hüllermeier.
        Link: https://arxiv.org/abs/2302.00736v3

    Note:
        Basic usage of explainer.

        >>> explainer = SSWARM(automl)
        >>> shap_values = explainer.shap_values(X_test, n_jobs=8)
        >>> shap.summary_plot(shap_values[0], X_test[explainer.used_feats])

    Args:
        automl: Automl object.
        random_state: Random seed for sampling combinations of features
            and column permutations.
    """

    def __init__(self, model, random_state: int = 77):
        self.model = model

        # keep only the used features
        self.used_feats = model.reader.used_features
        self.n = len(self.used_feats)
        self.N = set(np.arange(self.n))
        self.n_outputs = 1
        if self.model.reader._n_classes:
            self.n_outputs = self.model.reader._n_classes

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def shap_values(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: List[str] = None,
        T: int = None,
        n_repeats: int = 3,
        n_jobs: int = 1,
        batch_size: int = 500_000,
    ) -> List[List[float]]:
        """Computes shapley values of an input observations.

           Returns values in the SHAP format to be consistent with
           SHAP python library graphing tools.

        Args:
            data: A matrix of samples on which to explain the model's output.
            feature_names: Feature names for automl prediction when data is not pd.DataFrame .
            T: Number of iterations. Default is None, which refers to auto infer.
                Higher `T` yields more accurate estimates, but takes more time. Auto is ok in most cases.
            n_repeats: Number of computations with different seeds.
                Final shapley values are an avarage among seeds. Default is 3.
                Higher `n_repeats` yields more accurate estimates, but takes more time.
            n_jobs: Number of parallel workers to execute automl.predict() .
                IMPORTANT NOTE: for now, parallelization significantly increases
                the computational time of Shapley estimates for TabularAutoML models containing neural networks.
                Set the n_jobs parameter to 1 (the default value) if computing Shapley estimates
                for a model with a neural network algorithm.
            batch_size: Number of observations passed to automl.predict() at once.

        Returns:
            (# classes x # samples x # features) array of shapley values.
            (# samples x # features) in regression case.

        """
        # check for neural networks in model algorithms if n_jobs != 1
        if n_jobs != 1:
            for algo in list(self.model.collect_model_stats().keys()):
                if "TorchNN" in algo:
                    # model contains neural network
                    raise ValueError(
                        "Model contains neural network: {0}. ".format(algo)
                        + "Set the n_jobs parameter to 1. "
                        + "Read more in a docstring."
                    )

        # check if input number of observations is not less than 2
        num_obs = data.shape[0]
        if num_obs < 2:
            raise ValueError("Too small number of observations. " + "Input data must contatin at least 2 observations.")

        # check if input number of observations is more than a batch_size
        if num_obs > batch_size:
            raise ValueError("Decrease the input number of observations or increase the batch_size")

        feature_names = feature_names
        if isinstance(data, pd.DataFrame) and feature_names is None:
            feature_names = data.columns.values

        data = np.array(data)
        feature_names = np.array(feature_names)

        # infer auto params
        if T is None:
            if self.n < 20:
                T = 250
            else:
                T = 500

        # predictions on all features v(N)
        v_N = self.v(data, feature_names=feature_names, n_jobs=1)
        self.expected_value = np.mean(v_N, axis=0)

        # final phi is accumulated during repeats
        final_phi = np.zeros((data.shape[0], self.n, self.n_outputs))

        # initialization of \tilde{P}
        PMF = self.define_probability_distribution(self.n)
        if len(PMF) == 1:
            PMF = np.array([1])

        # if self.n > 3: # if n <= 3 -> we have already calculated all Shapley values exactly
        bar = tqdm(total=(2 * T * n_repeats))
        for k in range(n_repeats):

            # initializing arrays for main variables
            # size is (num_obs x n x n x num_outputs)
            self.phi_plus = np.zeros((data.shape[0], self.n, self.n, self.n_outputs))
            self.phi_minus = np.zeros((data.shape[0], self.n, self.n, self.n_outputs))
            self.c_plus = np.zeros((data.shape[0], self.n, self.n, self.n_outputs))
            self.c_minus = np.zeros((data.shape[0], self.n, self.n, self.n_outputs))

            # First stage: collect updates

            # Update is represented as a list of 3 sets
            # All updates are stored in a list
            self.updates = []

            # exact calculations of phi for coalitions of size 1, n-1, n
            self.exactCalculation()

            # warm ups
            self.warmUp("plus")
            self.warmUp("minus")

            # collect general updates
            for i in range(len(self.updates), T):
                # draw the size of coalition from distribution P
                s_t = self.rng.choice(np.arange(2, self.n - 1), p=PMF)

                # draw the random coalition with size s_t
                A_t = self.draw_random_combination(self.N, s_t)

                # store combination
                self.updates.append([A_t, A_t, self.N.difference(A_t)])

            # Second stage: make updates

            n_updates_per_round = min(batch_size // num_obs, T)
            for i in range(0, T, n_updates_per_round):

                # initialize array for shuffled data
                pred_data = np.empty((n_updates_per_round * num_obs, data.shape[1]), dtype=object)

                # prepare the data
                iter_updates = self.updates[i : i + n_updates_per_round]
                for j, comb in enumerate(iter_updates):
                    A = comb[0]
                    A_plus = comb[1]
                    A_minus = comb[2]

                    temp = copy(data)
                    for col in self.N.difference(A):
                        # map column number from the used features space to the overall features space
                        mapped_col = np.where(np.array(feature_names) == self.used_feats[col])[0][0]

                        # shuffle mapped column
                        temp[:, mapped_col] = self.rng.permutation(temp[:, mapped_col])

                    pred_data[j * num_obs : (j + 1) * num_obs] = temp
                    bar.update(n=1)

                # make predictions on shuffled data
                v = self.v(pred_data, feature_names=feature_names, n_jobs=n_jobs)

                # update phi-s
                for j, comb in enumerate(iter_updates):
                    A = comb[0]
                    A_plus = comb[1]
                    A_minus = comb[2]

                    v_t = v[j * num_obs : (j + 1) * num_obs]
                    self.update(A, A_plus, A_minus, v_t)

                    bar.update(n=1)

            # finilize the computations and derive the general representation of phi
            phi = np.sum(self.phi_plus - self.phi_minus, axis=2) / self.n
            PHI = np.sum(phi, axis=1)

            # normalize phi to sum to the predicted outcome and
            # substract expected value to be consistent with SHAP python library
            correction = ((v_N - PHI) / self.n).repeat(self.n, axis=0).reshape(len(phi), self.n, self.n_outputs)
            phi = phi + correction - self.expected_value / self.n

            # increment final phi
            final_phi += phi / n_repeats

            # change the seed
            self.rng = np.random.default_rng(seed=self.random_state + k + 1)

        bar.close()
        final_phi = np.transpose(final_phi, axes=[2, 0, 1])
        self.rng = np.random.default_rng(seed=self.random_state)

        if final_phi.shape[0] == 1:  # regression
            final_phi = final_phi[0]
            self.expected_value = self.expected_value[0]

        return final_phi

    def draw_random_combination(self, set_of_elements: set, size: int) -> Set:
        """Faster way of sampling a combination.

        Args:
            set_of_elements: set of features.
            size: size of the combination to be drawn.

        Returns:
            Random combination of features of size `size`.
        """
        combination = set()
        for _ in range(size):
            val = self.rng.choice(list(set_of_elements.difference(combination)))
            combination.add(val)
        return combination

    def v(self, data: np.array, feature_names: List[str] = None, n_jobs: int = 1) -> List[List[float]]:
        """Evaluate the utility function.

        Args:
            data: Data for prediction.
            n_jobs: Number of parallel workers to execute automl.predict() .
            feature_names: Feature names for automl.predict() .

        Returns:
            (# obs x # classes) array of predicted target.
        """
        if "n_jobs" in getfullargspec(self.model.predict).args:
            v = self.model.predict(data, features_names=feature_names, n_jobs=n_jobs).data
        else:
            v = self.model.predict(pd.DataFrame(data, columns=feature_names)).data

        if self.model.reader.task.name in ["reg", "multiclass"]:
            return v
        elif self.model.reader.task.name == "binary":
            return np.hstack((1 - v, v))
        else:
            raise NotImplementedError("Unknown task")

    def update(self, A, A_plus, A_minus, v):
        """Perform one update step."""
        card = len(A)
        for i in A_plus.intersection(A):
            phi_col = self.phi_plus[:, i, card - 1]
            c_col = self.c_plus[:, i, card - 1]

            self.phi_plus[:, i, card - 1] = (phi_col * c_col + v) / (c_col + 1)
            self.c_plus[:, i, card - 1] = c_col + 1

        for i in A_minus.difference(A):
            phi_col = self.phi_minus[:, i, card]
            c_col = self.c_minus[:, i, card]

            self.phi_minus[:, i, card] = (phi_col * c_col + v) / (c_col + 1)
            self.c_minus[:, i, card] = c_col + 1

    def exactCalculation(self):
        """Exact calculations."""
        for s in [1, self.n - 1, self.n]:
            for combination in combinations(self.N, s):
                combination = set(combination)
                inv_combination = self.N.difference(combination)
                self.updates.append([combination, combination, inv_combination])

    def warmUp(self, kind: str):
        """Warm ups."""
        for s in range(2, self.n - 1):
            pi = self.rng.permutation(self.n)
            for k in range(np.floor(self.n / s).astype(int)):
                A = set(pi[[i + k * s for i in range(0, s)]])
                if kind == "plus":
                    self.updates.append([A, A, set()])
                elif kind == "minus":
                    self.updates.append([self.N.difference(A), set(), A])

            if self.n % s != 0:
                A = set(pi[[self.n - i for i in range(1, self.n % s + 1)]])
                B = self.draw_random_combination(self.N.difference(A), s - (self.n % s))
                if kind == "plus":
                    self.updates.append([A.union(B), A, set()])
                elif kind == "minus":
                    self.updates.append([self.N.difference(A.union(B)), set(), A])

    @staticmethod
    def H(n: int) -> float:
        """n-th harmonic number."""
        return np.sum([1 / i for i in range(1, n + 1)])

    @staticmethod
    def define_probability_distribution(n: int) -> List[float]:
        """Compute P-tilde distribution.

        Args:
            n: number of features.

        Returns:
            array of PMF.
        """
        probas = np.empty(n - 3)
        if n % 2 == 0:  # even case
            for s in range(2, n - 1):
                if s <= (n - 2) // 2:
                    probas[s - 2] = (n * np.log(n) - 1) / (2 * s * n * np.log(n) * (SSWARM.H(n // 2 - 1) - 1))
                elif s == n // 2:
                    probas[s - 2] = 1 / (n * np.log(n))
                else:
                    probas[s - 2] = (n * np.log(n) - 1) / (2 * (n - s) * n * np.log(n) * (SSWARM.H(n // 2 - 1) - 1))
        else:  # odd case
            for s in range(2, n - 1):
                if s <= (n - 1) // 2:
                    probas[s - 2] = 1 / (2 * s * (SSWARM.H((n - 1) // 2) - 1))
                else:
                    probas[s - 2] = 1 / (2 * (n - s) * (SSWARM.H((n - 1) // 2) - 1))
        return probas
