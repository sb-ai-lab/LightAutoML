"""Basic classes for validation iterators."""

from copy import copy
from typing import Any
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.pipelines.features.base import FeaturesPipeline


# from ..pipelines.selection.base import SelectionPipeline

# TODO: SOLVE CYCLIC IMPORT PROBLEM!!! add Selectors typing

Dataset = TypeVar("Dataset", bound=LAMLDataset)
CustomIdxs = Iterable[Tuple[Sequence, Sequence]]


# add checks here
# check for same columns in dataset
class TrainValidIterator:
    """Abstract class to train/validation iteration.

    Train/valid iterator:
    should implement `__iter__` and `__next__` for using in ml_pipeline.

    """

    @property
    def features(self):
        """Dataset features names.

        Returns:
            List of features names.

        """
        return self.train.features

    def __init__(self, train: Dataset, **kwargs: Any):
        """

        Args:
            train: Train dataset.
            **kwargs: Key-word parameters.

        """
        self.train = train
        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    def __iter__(self) -> Iterable:
        """ Abstract method. Creates iterator."""
        raise NotImplementedError

    def __len__(self) -> Optional[int]:
        """Abstract method. Get length of dataset."""
        raise NotImplementedError

    def get_validation_data(self) -> LAMLDataset:
        """Abstract method. Get validation sample."""
        raise NotImplementedError

    def apply_feature_pipeline(self, features_pipeline: FeaturesPipeline) -> "TrainValidIterator":
        """Apply features pipeline on train data.

        Args:
            features_pipeline: Composite transformation of features.

        Returns:
            Copy of object with transformed features.

        """
        train_valid = copy(self)
        train_valid.train = features_pipeline.fit_transform(train_valid.train)
        return train_valid

    # TODO: add typing
    def apply_selector(self, selector) -> "TrainValidIterator":
        """Select features on train data.

        Check if selector is fitted.
        If not - fit and then perform selection.
        If fitted, check if it's ok to apply.

        Args:
            selector: Uses for feature selection.

        Returns:
            Dataset with selected features.

        """
        if not selector.is_fitted:
            selector.fit(self)
        train_valid = copy(self)
        train_valid.train = selector.select(train_valid.train)
        return train_valid

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Abstract method. Convert iterator to HoldoutIterator."""
        raise NotImplementedError


class DummyIterator(TrainValidIterator):
    """Simple Iterator which use train data as validation."""

    def __init__(self, train: Dataset):
        """Create iterator. WARNING: validation on train.

        Args:
            train: Train dataset.

        """
        self.train = train

    def __len__(self) -> Optional[int]:
        """Get 1 len.

        Returns:
            '1'.

        """
        return 1

    def __iter__(self) -> List[Tuple[None, Dataset, Dataset]]:
        """Simple iterable object.

        Returns:
            Iterable object for dataset, where for validation also uses train.

        """
        return [(None, self.train, self.train)]

    def get_validation_data(self) -> Dataset:
        """Just get validation sample.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Convert iterator to hold-out-iterator.

        Returns:
            iterator: Holdout iterator with ``'train == valid'``.

        """
        return HoldoutIterator(self.train, self.train)


class HoldoutIterator(TrainValidIterator):
    """Iterator for classic holdout - just predefined train and valid samples."""

    def __init__(self, train: LAMLDataset, valid: LAMLDataset):
        """Create iterator.

        Args:
            train: Dataset of train data.
            valid: Dataset of valid data.

        """
        self.train = train
        self.valid = valid

    def __len__(self) -> Optional[int]:
        """Get 1 len.

        Returns:
            1

        """
        return 1

    def __iter__(self) -> Iterable[Tuple[None, LAMLDataset, LAMLDataset]]:
        """Simple iterable object.

        Returns:
            Iterable object for train validation dataset.

        """
        return iter([(None, self.train, self.valid)])

    def get_validation_data(self) -> LAMLDataset:
        """Just get validation sample.

        Returns:
            Whole validation dataset.

        """
        return self.valid

    def apply_feature_pipeline(self, features_pipeline: FeaturesPipeline) -> "HoldoutIterator":
        """Inplace apply features pipeline to iterator components.

        Args:
            features_pipeline: Features pipeline to apply.

        Returns:
            New iterator.

        """
        train_valid = cast("HoldoutIterator", super().apply_feature_pipeline(features_pipeline))
        train_valid.valid = features_pipeline.transform(train_valid.valid)

        return train_valid

    def apply_selector(self, selector) -> "HoldoutIterator":
        """Same as for basic class, but also apply to validation.

        Args:
            selector: Uses for feature selection.

        Returns:
            New iterator.

        """
        train_valid = cast("HoldoutIterator", super().apply_selector(selector))
        train_valid.valid = selector.select(train_valid.valid)

        return train_valid

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Do nothing, just return itself.

        Returns:
            self.

        """
        return self


class CustomIterator(TrainValidIterator):
    """Iterator that uses function to create folds indexes.

    Usefull for example - classic timeseries splits.

    """

    def __init__(self, train: LAMLDataset, iterator: CustomIdxs):
        """Create iterator.

        Args:
            train: Dataset of train data.
            iterator: Callable(dataset) -> Iterator of train/valid indexes.

        """
        self.train = train
        self.iterator = iterator

    def __len__(self) -> Optional[int]:
        """Empty __len__ method.

        Returns:
            None.

        """

        return len(self.iterator)

    def __iter__(self) -> Generator:
        """Create generator of train/valid datasets.

        Returns:
            Data generator.

        """
        generator = ((val_idx, self.train[tr_idx], self.train[val_idx]) for (tr_idx, val_idx) in self.iterator)

        return generator

    def get_validation_data(self) -> LAMLDataset:
        """Simple return train dataset.

        Returns:
            Dataset of train data.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Convert iterator to hold-out-iterator.

        Use first train/valid split for :class:`~lightautoml.validation.base.HoldoutIterator` creation.

        Returns:
            New hold out iterator.

        """
        for (tr_idx, val_idx) in self.iterator:
            return HoldoutIterator(self.train[tr_idx], self.train[val_idx])
