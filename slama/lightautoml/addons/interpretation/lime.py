from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.linear_model import lars_path
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from ...pipelines.features.text_pipeline import _tokenizer_by_lang
from .utils import IndexedString
from .utils import draw_html


class TextExplanation:
    """Explanation of object for textual data."""

    def __init__(
        self,
        index_string: IndexedString,
        task_name: str,
        prediction: np.ndarray,
        class_names: Optional[List[Any]] = None,
        random_state=None,
        draw_prediction: bool = False,
    ):
        """

        Args:
            index_string: Pertrubing string object.
            task_name: Task name. Can be
                one of ['binary', 'multiclass', 'reg'].
            class_names: List of class names.
            random_state: Random seed for perturbation generation.

        """
        self.idx_str = index_string
        assert task_name in ["binary", "multiclass", "reg"]
        self.task_name = task_name
        self.prediction = prediction
        self.class_names = class_names
        self.draw_prediction = draw_prediction

        if task_name == "reg":
            self.default_label = 0
        else:
            self.default_label = 1

        self.random_state = check_random_state(random_state)
        self.instance = {}

    def as_list(self, label: Optional[int] = None) -> List[Tuple[int, float]]:
        """Get feature weights as list.

        Args:
            label: Explaing label. Not necessary
                for regression. By default, for
                regression 0 will be used, and 1 for
                other task types.

        Returns:
            List in format (token_id, weight).

        """
        label = self._label(label)
        ans = self.instance[label]["feature_weights"]
        ans = [(x[0], float(x[1])) for x in ans]

        return ans

    def as_features(
        self,
        label: Optional[int] = None,
        add_not_rel: bool = False,
        normalize: bool = False,
    ) -> List[Tuple[str, float]]:
        """Get feature weights as list with feature names.

        Args:
            label: Explaing label. Not necessary
                for regression. By default, for
                regression 0 will be used, and 1 for
                other task types.

            add_not_rel: The not relevalt tokens will be
                added in explanation with zero feature weights.
                Using this flag it is convenient to visualize
                the predictions.

            normalize: Normalization by the maximum
                absolute value of the importance of the feature.

        Returns:
            List in format (token, weight).

        """
        label = self._label(label)
        fw = self.instance[label]["feature_weights"]
        norm_const = 1.0
        if normalize:
            norm_const = 1 / abs(fw[0][1])
        fw = dict(fw)
        if add_not_rel:
            fw = dict(self.instance[label]["feature_weights"])
            weights = np.zeros_like(self.idx_str.as_np_, dtype=np.float32)
            for k, v in fw.items():
                weights[self.idx_str.pos[k]] = v

            ans = [(k, float(w) * norm_const) for k, w in zip(self.idx_str.as_np_, weights)]
        else:
            ans = [(self.idx_str.word(k), float(v) * norm_const) for k, v in fw.items()]

        return ans

    def as_map(self, label: Optional[int] = None) -> Dict[str, float]:
        """Get feature weights as list with features.

        Args:
            label: Explaing label. Not necessary
                for regression. By default, for
                regression 0 will be used, and 1 for
                other task types.

        Returns:
            Dictonary of tokens and it's weights in format
            ({token_id}_{position}, feature_weight).

        """
        label = self._label(label)
        return {f"{k}_{i}": v for i, (k, v) in enumerate(self.instance[label]["feature_weights"])}

    def as_html(self, label: Optional[int] = None) -> str:
        """Generates inline HTML with colors.

        Args:
            label: Explaing label. Not necessary
                for regression. By default, for
                regression 0 will be used, and 1 for
                other task types.

        Returns:
            HTML code.

        """

        label = self._label(label)
        weight_string = self.as_features(label, add_not_rel=True, normalize=False)
        prediction = self.prediction[label]
        if not self.draw_prediction:
            prediction = None

        if self.task_name == "reg":
            return draw_html(weight_string, self.task_name, prediction=prediction)
        elif self.task_name == "binary":
            return draw_html(
                weight_string,
                self.task_name,
                prediction=prediction,
                grad_line=True,
                grad_positive_label=str(self.class_names[label]),
                grad_negative_label=str(self.class_names[1 - label]),
            )
        elif self.task_name == "multiclass":
            return draw_html(
                weight_string,
                self.task_name,
                prediction=prediction,
                grad_line=True,
                grad_positive_label=str(self.class_names[label]),
            )

    def visualize_in_notebook(self, label: Optional[int] = None):
        """Visualization of interpretation in IPython notebook.

        Args:
            label: Explaing label. Not necessary
                for regression. By default, for
                regression 0 will be used, and 1 for
                other task types.

        """
        from IPython.display import HTML
        from IPython.display import display_html

        label = self._label(label)
        raw_html = self.as_html(label)
        display_html(HTML(raw_html))

    def _label(self, label: Union[None, int]) -> int:
        if label is None or self.task_name == "reg":
            label = self.default_label

        return label

    def get_label(self, name):
        return self.class_names.index(name)


class LimeTextExplainer:
    """Instance-wise textual explanation.

    Method working as follows:
    1. Tokenize perturbed text-column.
    2. Create dataset by perturbing text column.
    3. Select features (Optional).
    4. Fit explainable model (distil_model). (For multiclass
    one-versus-all will be used)
    5. Get the explanation from distil_model.

    Note:
        More info: `"Why Should I Trust You?":
        Explaining the Predictions of Any Classifier
        Ribeiro et al. <https://arxiv.org/abs/1602.04938>`_

    Note:
        Basic usage of explaier.

        >>> task = Task('reg')
        >>> automl = TabularNLPAutoML(task=task,
        >>>     timeout=600, gpu_ids = '0',
        >>>     general_params = {'nested_cv': False, 'use_algos': [['nn']]},
        >>>     text_params = {'lang': 'ru'})
        >>> automl.fit_predict(train, roles=roles)
        >>> lime = LimeTextExplainer(automl)
        >>> explanation = lime.explain_instance(train.iloc[0], perturb_column='message')
        >>> explanation.visualize_in_notebook()

    """

    def __init__(
        self,
        automl,
        kernel: Optional[Callable] = None,
        kernel_width: float = 25.0,
        feature_selection: str = "none",
        force_order: bool = False,
        model_regressor: Any = None,
        distance_metric: str = "cosine",
        random_state: Union[int, np.random.RandomState] = 0,
        draw_prediction: bool = False,
    ):
        """

        Args:
            automl: Automl object.
            kernel: Callable object with parameter `kernel_width`.
                By default, the squared-exponential kernel will be used.
            kernel_width: Kernel width.
            feature_selection: Feature selection type. For now,
                'none', 'lasso' are availiable.
            force_order: Whether to follow the word order.
            model_regressor: Model distilator. By default,
                Ridge regression will be used.
            distance_metric: Distance type between binary vectors.
            random_state: Random seed used,
                for sampling perturbation of text column.

        """
        self.automl = automl
        self.task_name = automl.reader.task.name

        assert self.task_name in ["binary", "multiclass", "reg"]

        self.roles = automl.reader.roles
        self.kernel_width = kernel_width
        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        assert callable(kernel)

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.random_state = check_random_state(random_state)

        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)

        self.distil_model = model_regressor

        # todo: forward selection and higher weights
        # and l2x
        # and auto-mode

        assert feature_selection in ["none", "lasso"]

        self.feature_selection = feature_selection
        self.force_order = force_order

        lang = automl.text_params["lang"]
        self.tokenizer = _tokenizer_by_lang[lang](is_stemmer=False)
        self.distance_metric = distance_metric

        class_names = automl.reader.class_mapping
        if class_names is None:
            if self.task_name == "reg":
                class_names = [0]
            else:
                class_names = np.arange(automl.reader._n_classes)
        else:
            class_names = list(class_names.values())

        self.class_names = class_names
        self.draw_prediction = draw_prediction

    def explain_instance(
        self,
        data: pd.Series,
        perturb_column: str,
        labels: Optional[Iterable] = None,
        n_features: int = 10,
        n_samples: int = 5000,
    ) -> "TextExplanation":
        """

        Args:
            data: Data sample to explain.
            perturb_column: Column that will be perturbed.
            labels: Target variable values to be interpreted.
            n_features: If a feature selector was specified,
                then this number means the maximum number
                of features that will be used in the distilled model.
            n_samples: Number of sampled instances by perturbing text column.

        Returns:
            TextExplanation object.

        """
        assert self.roles[perturb_column].name == "Text", "Column is not text column"
        assert n_samples > 1, "Number of generated samples must be > 0"

        if labels is None:
            labels = (0,) if self.task_name == "reg" else (1,)

        data, y, dst, expl = self._get_perturb_dataset(data, perturb_column, n_samples)

        for label in labels:
            expl.instance[label] = self._explain_dataset(data, y, dst, label, n_features)

        return expl

    def _get_perturb_dataset(self, data, perturb_column: str, n_samples: int):
        text = data[perturb_column]
        idx_str = IndexedString(text, self.tokenizer, self.force_order)
        n_words = idx_str.n_words
        samples = self.random_state.randint(1, n_words + 1, n_samples - 1)
        raw_dataset = [data.copy()]
        dataset = np.ones((n_samples, n_words))

        for i, size in enumerate(samples, start=1):
            off_tokens = self.random_state.choice(range(n_words), size)
            data_ = data.copy()
            p_text = idx_str.inverse_removing(off_tokens)
            data_[perturb_column] = p_text
            raw_dataset.append(data_)
            dataset[i, off_tokens] = 0

        raw_dataset = pd.DataFrame(raw_dataset)

        pred = self.automl.predict(raw_dataset).data
        if self.task_name == "binary":
            pred = np.concatenate([1 - pred, pred], axis=1)

        distance = pairwise_distances(dataset, dataset[0].reshape(1, -1), metric=self.distance_metric).ravel()

        expl = TextExplanation(idx_str, self.task_name, pred[0], self.class_names, self.random_state)

        return dataset, pred, distance * 100, expl

    def _explain_dataset(
        self,
        data: pd.DataFrame,
        y: np.array,
        dst: np.array,
        label: int,
        n_features: int,
    ) -> Dict[str, Union[float, np.array]]:
        weights = self.kernel_fn(dst)
        y = y[:, label]
        features = self._feature_selection(data, y, weights, n_features, mode=self.feature_selection)
        model = self.distil_model
        model.fit(data[:, features], y, sample_weight=weights)
        score = model.score(data[:, features], y, sample_weight=weights)

        pred = model.predict(data[0, features].reshape(1, -1))
        feature_weights = list(sorted(zip(features, model.coef_), key=lambda x: np.abs(x[1]), reverse=True))
        res = {
            "bias": model.intercept_,
            "feature_weights": feature_weights,
            "score": score,
            "pred": pred,
        }

        return res

    def _feature_selection(
        self,
        data: pd.DataFrame,
        y: np.array,
        weights: np.array,
        n_features: int,
        mode: str = "none",
    ) -> List[int]:
        if mode == "none":
            return np.arange(data.shape[1])
        if mode == "lasso":
            weighted_data = (data - np.average(data, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis])
            weighted_y = (y - np.average(y, weights=weights)) * np.sqrt(weights)

            features = np.arange(weighted_data.shape[1])
            _, _, coefs = lars_path(weighted_data, weighted_y, method="lasso", verbose=False)

            for i in range(len(coefs.T) - 1, 0, -1):
                features = coefs.T[i].nonzero()[0]
                if len(features) <= n_features:
                    break

            return features
