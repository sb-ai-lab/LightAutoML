"""SSWARM."""

from inspect import getfullargspec
from itertools import combinations
from typing import List
from typing import Set
from typing import Union

import numpy as np
import pandas as pd

from tqdm import tqdm


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
            T: Total coalition evaluations per repeat, including initialization.
                Must cover all exact and warm-up updates. None selects at least
                250 evaluations for fewer than 20 features, or 500 otherwise,
                increasing this budget to cover initialization when necessary.
                With at most three features, all nonempty coalitions are used
                once per repeat, even if a larger budget is supplied.
            n_repeats: Number of computations with different seeds.
                Final shapley values are an average among seeds. Default is 3.
                Higher `n_repeats` yields more accurate estimates, but takes more time.
            n_jobs: Number of parallel workers to execute automl.predict() .
                IMPORTANT NOTE: for now, parallelization significantly increases
                the computational time of Shapley estimates for TabularAutoML models containing neural networks.
                Set the n_jobs parameter to 1 (the default value) if computing Shapley estimates
                for a model with a neural network algorithm.
            batch_size: Maximum number of observations passed to predict at once.
                Must be at least the number of input observations. Lower this
                value to reduce prediction memory without changing sampling.

        Returns:
            (# classes x # samples x # features) array of shapley values.
            (# samples x # features) in regression case.

        Raises:
            ValueError: If the input, feature names or sampling budget is invalid.

        Note:
            Accumulators use O(samples * features * outputs + features ** 2)
            memory. The coalition schedule and prediction batches need additional
            memory. Parallel prediction workers can also increase memory use.

        """
        # check for neural networks in model algorithms if n_jobs != 1
        if n_jobs != 1:
            for algo in list(self.model.collect_model_stats().keys()):
                if "TorchNN" in algo:
                    # model contains neural network
                    raise ValueError(
                        f"Model contains neural network: {algo}. "
                        + "Set the n_jobs parameter to 1. "
                        + "Read more in a docstring."
                    )

        for name, value in (("T", T), ("n_repeats", n_repeats), ("batch_size", batch_size)):
            if name == "T" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n == 0:
            raise ValueError("The model must use at least one feature")
        if data.ndim != 2:
            raise ValueError("Input data must be a two-dimensional matrix")

        num_obs = data.shape[0]
        if num_obs < 2:
            raise ValueError("Too small number of observations. " + "Input data must contain at least 2 observations.")

        # check if input number of observations is more than a batch_size
        if num_obs > batch_size:
            raise ValueError("Decrease the input number of observations or increase the batch_size")

        if isinstance(data, pd.DataFrame) and feature_names is None:
            feature_names = data.columns
        if feature_names is None or len(feature_names) != data.shape[1]:
            raise ValueError("feature_names must contain one name per input column")
        feature_names = pd.Index(feature_names)
        if not feature_names.is_unique:
            raise ValueError("feature_names must be unique")
        columns = feature_names.get_indexer(self.used_feats)
        if (columns < 0).any():
            missing = [name for name, col in zip(self.used_feats, columns) if col < 0]
            raise ValueError(f"Missing model features: {missing}")

        # Select before converting: unused object columns must not box numeric data.
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, columns].to_numpy()
        else:
            data = np.asarray(data)[:, columns]
        feature_names = self.used_feats

        if self.n <= 3:
            minimum = 2 ** self.n - 1
        else:
            minimum = 2 * self.n + 1 + 2 * sum((self.n + s - 1) // s for s in range(2, self.n - 1))
        if T is not None and T < minimum:
            raise ValueError(f"T must be at least {minimum} to initialize all strata for {self.n} features")
        if self.n <= 3:
            T = minimum
        elif T is None:
            T = max(250 if self.n < 20 else 500, minimum)

        # predictions on all features v(N)
        v_N = self.v(data, feature_names=feature_names, n_jobs=1)
        self.expected_value = np.mean(v_N, axis=0)

        # final phi is accumulated during repeats
        final_phi = np.zeros((data.shape[0], self.n, self.n_outputs))

        PMF = self.define_probability_distribution(self.n) if self.n > 3 else None
        if self.n == 4:
            PMF = np.array([1.0])

        bar = tqdm(total=2 * T * n_repeats)
        try:
            self._phi = np.zeros_like(final_phi)
            self.c_plus = np.zeros((self.n, self.n), dtype=np.int64)
            self.c_minus = np.zeros_like(self.c_plus)
            n_updates_per_round = min(batch_size // num_obs, T)
            pred_data = np.empty((n_updates_per_round * num_obs, self.n), dtype=data.dtype)

            for k in range(n_repeats):
                self._phi.fill(0)
                self.c_plus.fill(0)
                self.c_minus.fill(0)
                self.updates = []
                self.exactCalculation()
                self.warmUp("plus")
                self.warmUp("minus")
                for _ in range(len(self.updates), T):
                    s_t = self.rng.choice(np.arange(2, self.n - 1), p=PMF)
                    A_t = self.draw_random_combination(self.N, s_t)
                    self.updates.append([A_t, A_t, self.N.difference(A_t)])

                # Counts depend only on the schedule, not on observations or outputs.
                # Knowing the final denominator lets each prediction contribute to
                # the sum of stratum means without storing individual means.
                for A, A_plus, A_minus in self.updates:
                    card = len(A)
                    for feature in A_plus.intersection(A):
                        self.c_plus[feature, card - 1] += 1
                    for feature in A_minus.difference(A):
                        self.c_minus[feature, card] += 1

                for i in range(0, T, n_updates_per_round):
                    iter_updates = self.updates[i : i + n_updates_per_round]
                    batch = pred_data[: len(iter_updates) * num_obs]
                    for j, (A, _, _) in enumerate(iter_updates):
                        shuffled = batch[j * num_obs : (j + 1) * num_obs]
                        shuffled[:] = data
                        for col in self.N.difference(A):
                            shuffled[:, col] = self.rng.permutation(data[:, col])
                        bar.update(n=1)

                    v = self.v(batch, feature_names=feature_names, n_jobs=n_jobs)
                    for j, (A, A_plus, A_minus) in enumerate(iter_updates):
                        self.update(A, A_plus, A_minus, v[j * num_obs : (j + 1) * num_obs])
                        bar.update(n=1)
                    del v

                self._phi /= self.n
                correction = (v_N - self._phi.sum(axis=1)) / self.n
                self._phi += correction[:, None, :]
                self._phi -= self.expected_value / self.n
                self._phi /= n_repeats
                final_phi += self._phi
                self.rng = np.random.default_rng(seed=self.random_state + k + 1)
        finally:
            # Do not keep working buffers alive on the explainer, even after failure.
            for name in ("_phi", "c_plus", "c_minus", "updates"):
                self.__dict__.pop(name, None)
            self.rng = np.random.default_rng(seed=self.random_state)
            bar.close()

        final_phi = np.transpose(final_phi, axes=[2, 0, 1])

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
        """Accumulate one prediction using the final stratum counts."""
        card = len(A)
        v = np.asarray(v, dtype=np.float64)
        for i in A_plus.intersection(A):
            self._phi[:, i] += v / self.c_plus[i, card - 1]

        for i in A_minus.difference(A):
            self._phi[:, i] -= v / self.c_minus[i, card]

    def exactCalculation(self):
        """Exact calculations."""
        for s in sorted({1, self.n - 1, self.n} - {0}):
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
