from typing import Union
from typing import List
from typing import Set

import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from copy import copy


class SSWARM:
    """Fast computation of shapley values.

    Base on the Stratified SWARM algorithm.
    Origin:
        Title: Approximating the Shapley Value without Marginal Contributions.
        Authors: Patrick Kolpaczki and Viktor Bengs and Maximilian Muschalik and Eyke Hüllermeier.
        Link: https://arxiv.org/abs/2302.00736v3

    Note:
        Basic usage of explaier.

        >>> explainer = SSWARM(automl, random_state=RANDOM_STATE)
        >>> shap_values = explainer.shap_values(X_test, n_jobs=N_THREADS)
        >>> shap.summary_plot(shap_values[0], X_test)

    Args:
        automl: Automl object.
        random_state: Random seed for sampling combinations of features.


    """

    def __init__(self, model, random_state: int = 77):
        self.model = model
        self.n_outputs = 1
        if self.model.reader._n_classes:
            self.n_outputs = self.model.reader._n_classes
        self.rng = np.random.default_rng(seed=random_state)

    def shap_values(
        self, data: Union[pd.DataFrame, np.array], feature_names: List[str] = None, T: int = 500, n_jobs: int = 1
    ) -> List[List[float]]:
        """Computes shapley values consistent with the SHAP interface.

        Args:
            data: A matrix of samples on which to explain the model's output.
            feature_names: Feature names for automl prediction when data is not pd.DataFrame .
            T: Number of iterations.
            n_jobs: Number of parallel workers to execute automl.predict() .

        Returns:
            (# classes x # samples x # features) array of shapley values.
            (# samples x # features) in regression case.

        """
        self.num_obs = data.shape[0]
        if self.num_obs < 2:
            raise ValueError(
                """Too small number of observations.
                             Input data must contatin at least 2 observations."""
            )
        self.n = data.shape[1]
        self.N = set(np.arange(data.shape[1]))

        self.feature_names = feature_names
        if isinstance(data, pd.DataFrame) and feature_names is None:
            self.feature_names = data.columns.values
        self.data = np.array(data)

        # store prediction on all features v(N)
        v_N = self.v(data, n_jobs=1)
        self.expected_value = np.mean(v_N, axis=0)

        # initializing arrays for main variables
        # size is (num_obs x n x n x num_outputs)
        self.phi_plus = np.zeros((data.shape[0], data.shape[1], data.shape[1], self.n_outputs))
        self.phi_minus = np.zeros((data.shape[0], data.shape[1], data.shape[1], self.n_outputs))
        self.c_plus = np.zeros((data.shape[0], data.shape[1], data.shape[1], self.n_outputs))
        self.c_minus = np.zeros((data.shape[0], data.shape[1], data.shape[1], self.n_outputs))

        # initialization of \tilde{P}
        PMF = self.define_probability_distribution(self.n)
        if len(PMF) == 1:
            PMF = np.array([1])

        # First stage: collect updates
        # Updates are stored as a list of 3 sets
        self.updates = []

        # exact calculations of phi for coalitions of size 1, n-1, n
        self.exactCalculation()

        # warm up stages
        self.warmUp("plus")
        self.warmUp("minus")

        t = len(self.updates)

        # collect general updates
        for i in range(t, T):
            # draw the size of coalition from distribution P
            s_t = self.rng.choice(np.arange(2, self.n - 1), p=PMF)
            # draw the random coalition with size s_t
            # A_t = set(self.rng.choice([list(i) for i in combinations(self.N, s_t)]))
            A_t = self.draw_random_combination(self.N, s_t)
            # store combination
            self.updates.append([A_t, A_t, self.N.difference(A_t)])

        # Second stage: make updates

        # Num of the predictions made at once
        batch_size = 500_000
        if self.num_obs > batch_size:
            raise ValueError("Decrease the input number of observations")

        n_updates_per_round = batch_size // self.num_obs
        bar = tqdm(total=2 * T)
        for i in range(0, T, n_updates_per_round):
            pred_data = np.empty((n_updates_per_round * self.num_obs, self.n), dtype=np.object)

            # prepare the data
            iter_updates = self.updates[i : i + n_updates_per_round]
            for j, comb in enumerate(iter_updates):
                A = comb[0]
                A_plus = comb[1]
                A_minus = comb[2]

                temp = copy(self.data)
                for col in self.N.difference(A):
                    temp[:, col] = self.rng.permutation(temp[:, col])

                pred_data[j * self.num_obs : (j + 1) * self.num_obs] = temp
                bar.update(n=1)

            # make predictions
            v = self.v(pred_data, n_jobs=n_jobs)

            # make updates
            for j, comb in enumerate(iter_updates):
                A = comb[0]
                A_plus = comb[1]
                A_minus = comb[2]

                v_t = v[j * self.num_obs : (j + 1) * self.num_obs]
                self.update(A, A_plus, A_minus, v_t)
                bar.update(n=1)

        bar.close()

        phi = np.sum(self.phi_plus - self.phi_minus, axis=2) / self.n
        PHI = np.sum(phi, axis=1)

        # normalize phi to sum to the predicted outcome
        # and substract expected value to be consistent with SHAP python library
        correction = ((v_N - PHI) / self.n).repeat(self.n, axis=0).reshape(len(phi), self.n, self.n_outputs)
        phi = phi + correction - self.expected_value / self.n

        phi = np.transpose(phi, axes=[2, 0, 1])
        if phi.shape[0] == 1:  # regression
            phi = phi[0]
            self.expected_value = self.expected_value[0]
        return phi

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

    def v(self, data: np.array, n_jobs: int = 1) -> List[List[float]]:
        """Evaluate the value function.

        Args:
            data: Data for prediction.
            n_jobs: Number of parallel workers to execute automl.predict() .

        Returns:
            (# obs x # classes) array of predicted target.
        """
        if isinstance(data, pd.DataFrame):
            v = self.model.predict(data, n_jobs=n_jobs).data
        else:
            v = self.model.predict(data, features_names=self.feature_names, n_jobs=n_jobs).data

        if self.model.task.name in ["reg", "multiclass"]:
            return v
        elif self.model.task.name == "binary":
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
        """Exact calculation stage."""
        for s in [1, self.n - 1, self.n]:
            for combination in combinations(self.N, s):
                combination = set(combination)
                inv_combination = self.N.difference(combination)
                self.updates.append([combination, combination, inv_combination])

    def warmUp(self, kind: str):
        """Warm up stage."""
        for s in range(2, self.data.shape[1] - 1):
            pi = self.rng.permutation(self.data.shape[1])
            for k in range(np.floor(self.data.shape[1] / s).astype(int)):
                A = set(pi[[i + k * s for i in range(0, s)]])
                if kind == "plus":
                    self.updates.append([A, A, set()])
                elif kind == "minus":
                    self.updates.append([self.N.difference(A), set(), A])

            if self.n % s != 0:
                A = set(pi[[self.n - i for i in range(1, self.n % s + 1)]])
                # B = set(self.rng.choice(
                #     [list(i) for i in combinations(self.N.difference(A), s - (self.n % s))]))
                B = self.draw_random_combination(self.N.difference(A), s - (self.n % s))
                if kind == "plus":
                    self.updates.append([A.union(B), A, set()])
                elif kind == "minus":
                    self.updates.append([self.N.difference(A.union(B)), set(), A])

    @staticmethod
    def H(n: int) -> float:
        """n-th harmonic number"""
        return np.sum([1 / i for i in range(1, n + 1)])

    @staticmethod
    def define_probability_distribution(n: int) -> List[float]:
        """Compute P-tilde distribution.

        Args:
            n: num of features.

        Returns:
            PMF.
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
