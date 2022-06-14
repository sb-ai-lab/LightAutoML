"""Base class for selection pipelines."""

from copy import copy
from copy import deepcopy
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from pandas import Series

from lightautoml.validation.base import TrainValidIterator

from ...dataset.base import LAMLDataset
from ...ml_algo.base import MLAlgo
from ...ml_algo.tuning.base import DefaultTuner
from ...ml_algo.tuning.base import ParamsTuner
from ...ml_algo.utils import tune_and_fit_predict
from ..features.base import FeaturesPipeline
from ..utils import map_pipeline_names


class ImportanceEstimator:
    """
    Abstract class, that estimates feature importances.
    """

    def __init__(self):
        self.raw_importances = None

    # Change signature here to be compatible with MLAlgo
    def fit(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def get_features_score(self) -> Series:
        """Get raw features importances.

        Returns:
            Pandas Series object with index - str features names and values - array of importances.

        """
        return self.raw_importances


class SelectionPipeline:
    """
    Abstract class, performing feature selection.
    Instance should accept train/valid datasets and select features.

    """

    @property
    def is_fitted(self) -> bool:
        """Check if selection pipeline is already fitted.

        Returns:
            ``True`` for fitted pipeline and False for not fitted.

        """
        return self._selected_features is not None

    @property
    def selected_features(self) -> List[str]:
        """Get selected features.

        Returns:
            List of selected feature names.

        """
        assert self._selected_features is not None, "Should be fitted first"
        return self._selected_features

    @selected_features.setter
    def selected_features(self, val: List[str]):
        """Setter of selected features.

        Args:
            val: List of selected feature names.

        """
        self._selected_features = deepcopy(val)

    @property
    def in_features(self) -> List[str]:
        """Input features to the selector.

        Raises exception if not fitted beforehand.

        Returns:
            List of input features.

        """
        assert self._in_features is not None, "Should be fitted first"
        return self._in_features

    @property
    def dropped_features(self) -> List[str]:
        """Features that were dropped.

        Returns:
            list of dropped features.

        """
        included = set(self._selected_features)
        return [x for x in self._in_features if x not in included]

    def __init__(
        self,
        features_pipeline: Optional[FeaturesPipeline] = None,
        ml_algo: Optional[Union[MLAlgo, Tuple[MLAlgo, ParamsTuner]]] = None,
        imp_estimator: Optional[ImportanceEstimator] = None,
        fit_on_holdout: bool = False,
        **kwargs: Any
    ):
        """Create features selection pipeline.

        Args:
            features_pipeline: Composition of feature transforms.
            ml_algo: Tuple (MlAlgo, ParamsTuner).
            imp_estimator: Feature importance estimator.
            fit_on_holdout: If use the holdout iterator.
            **kwargs: Not used.

        """
        self.features_pipeline = features_pipeline
        self._fit_on_holdout = fit_on_holdout

        self.ml_algo = None
        self._empty_algo = None
        if ml_algo is not None:
            try:
                self.ml_algo, self.tuner = ml_algo
            except (TypeError, ValueError):
                self.ml_algo, self.tuner = ml_algo, DefaultTuner()

            if not self.ml_algo.is_fitted:
                self._empty_algo = deepcopy(self.ml_algo)

        self.imp_estimator = imp_estimator
        self._selected_features = None
        self._in_features = None
        self.mapped_importances = None

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        """Select features from train-valid iterator.

        Method is used to perform selection based
        on features pipeline and ml model.
        Should save ``_selected_features`` attribute in the end of working.

        Raises:
            NotImplementedError.

        """
        raise NotImplementedError

    def fit(self, train_valid: TrainValidIterator):
        """Selection pipeline fit.

        Find features selection for given dataset based
        on features pipeline and ml model.

        Args:
            train_valid: Dataset iterator.

        """
        if not self.is_fitted:

            if self._fit_on_holdout:
                train_valid = train_valid.convert_to_holdout_iterator()

            self._in_features = train_valid.features
            if self.features_pipeline is not None:
                train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

            preds = None
            if self.ml_algo is not None:
                if self.ml_algo.is_fitted:
                    assert list(self.ml_algo.features) == list(
                        train_valid.features
                    ), "Features in feated MLAlgo should match exactly"
                else:
                    self.ml_algo, preds = tune_and_fit_predict(self.ml_algo, self.tuner, train_valid)

            if self.imp_estimator is not None:
                self.imp_estimator.fit(train_valid, self.ml_algo, preds)

            self.perform_selection(train_valid)

    def select(self, dataset: LAMLDataset) -> LAMLDataset:
        """Takes only selected features from giving dataset and creates new dataset.

        Args:
            dataset: Dataset for feature selection.

        Returns:
            New dataset with selected features only.

        """
        selected_features = copy(self.selected_features)
        # Add features that forces input
        sl_set = set(selected_features)
        roles = dataset.roles
        for col in (x for x in dataset.features if x not in sl_set):
            if roles[col].force_input:
                if col not in sl_set:
                    selected_features.append(col)

        return dataset[:, self.selected_features]

    def map_raw_feature_importances(self, raw_importances: Series):
        """Calculate input feature importances.
        Calculated as sum of importances on different levels of pipeline.

        Args:
            raw_importances: Importances of output features.

        """
        if self.features_pipeline is None:
            return raw_importances.copy()
        mapped = map_pipeline_names(self.in_features, raw_importances.index)
        mapped_importance = Series(raw_importances.values, index=mapped)

        self.mapped_importances = mapped_importance.groupby(level=0).sum().sort_values(ascending=False)

    def get_features_score(self):
        """Get input feature importances.

        Returns:
            Series with importances in not ascending order.

        """
        return self.mapped_importances


class EmptySelector(SelectionPipeline):
    """Empty selector - perform no selection, just save input features names."""

    def __init__(self):
        super().__init__()

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        """Just save input features names.

        Args:
            train_valid: Used for getting features names.

        """
        self._selected_features = train_valid.features


class PredefinedSelector(SelectionPipeline):
    """Predefined selector - selects columns specified by user."""

    def __init__(self, columns_to_select: Sequence[str]):
        """

        Args:
            columns_to_select: Columns will be selected.

        """
        super().__init__()
        self.columns_to_select = set(columns_to_select)

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        """Select only specified columns.

        Args:
            train_valid: Used for validation of features presence.

        """
        assert len(self.columns_to_select) == len(
            self.columns_to_select.intersection(set(train_valid.features))
        ), "Columns to select not match with dataset features"
        self._selected_features = list(self.columns_to_select)


class ComposedSelector(SelectionPipeline):
    """Composed selector - perform composition of selections."""

    def __init__(self, selectors: Sequence[SelectionPipeline]):
        """

        Args:
            selectors: Sequence of selectors.

        """
        super().__init__()
        self.selectors = selectors

    def fit(self, train_valid: Optional[TrainValidIterator] = None):
        """Fit all selectors in composition.

        Args:
            train_valid: Dataset iterator.

        """
        for selector in self.selectors:
            train_valid = train_valid.apply_selector(selector)

        self._in_features = self.selectors[0].in_features
        self.perform_selection(train_valid)

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        """Defines selected features.

        Args:
            train_valid: Not used.

        """
        self._selected_features = self.selectors[-1].selected_features

    def get_features_score(self):
        """Get mapped input features importances."""
        return self.selectors[-1].mapped_importances
