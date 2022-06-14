"""Basic classes for transformers."""

from copy import deepcopy
from typing import Callable
from typing import ClassVar
from typing import List
from typing import Sequence
from typing import Union

import numpy as np

from ..dataset.base import LAMLDataset
from ..dataset.base import RolesDict
from ..dataset.roles import ColumnRole
from ..dataset.utils import concatenate


# TODO: From func transformer

Roles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]


class LAMLTransformer:
    """Base class for transformer method (like sklearn, but works with datasets)."""

    _fname_prefix = None
    _fit_checks = ()
    _transform_checks = ()

    @property
    def features(self) -> List[str]:
        """Get name of the features, that will be generated after transform.

        Returns:
            List of new names.

        """
        if "_features" not in self.__dict__:
            raise AttributeError("Should be fitted at first.")

        feats = [
            "{0}__{1}".format(self._fname_prefix, x) if self._fname_prefix is not None else x for x in self._features
        ]

        return feats

    @features.setter
    def features(self, val: Sequence[str]):
        """Write input feature names.

        Args:
            val: Sequence of input features names.

        """
        self._features = deepcopy(val)

    def fit(self, dataset: LAMLDataset) -> "LAMLTransformer":
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        self.features = dataset.features
        for check_func in self._fit_checks:
            check_func(dataset)
        return self

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Transform on dataset.

        Args:
            dataset: Dataset to make transform.

        Returns:
            LAMLDataset with new features.

        """
        for check_func in self._transform_checks:
            check_func(dataset)
        return dataset

    def fit_transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Default implementation of fit_transform - fit and then transform.

        Args:
            dataset: Dataset to fit and then transform on it.

        Returns:
            Dataset with new features.

        """
        for check_func in self._fit_checks:
            check_func(dataset)
        self.fit(dataset)

        for check_func in self._transform_checks:
            check_func(dataset)

        return self.transform(dataset)


class SequentialTransformer(LAMLTransformer):
    """
    Transformer that contains the list of transformers and apply one by one sequentially.
    """

    def __init__(self, transformer_list: Sequence[LAMLTransformer]):
        """

        Args:
            transformer_list: Sequence of transformers.

        """
        self.transformer_list = transformer_list

    def fit(self, dataset: LAMLDataset):
        """Fit not supported. Needs output to fit next transformer.

        Args:
            dataset: Dataset to fit.

        """
        raise NotImplementedError("Sequential supports only fit_transform.")

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Apply the sequence of transformers to dataset one over output of previous.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        for trf in self.transformer_list:
            dataset = trf.transform(dataset)

        return dataset

    def fit_transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Sequential ``.fit_transform``.

         Output features - features from last transformer with no prefix.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        for trf in self.transformer_list:
            dataset = trf.fit_transform(dataset)

        self.features = self.transformer_list[-1].features

        return dataset


class UnionTransformer(LAMLTransformer):
    """Transformer that apply the sequence on transformers in parallel on dataset and concatenate the result."""

    def __init__(self, transformer_list: Sequence[LAMLTransformer], n_jobs: int = 1):
        """

        Args:
            transformer_list: Sequence of transformers.
            n_jobs: Number of processes to run fit and transform.

        """
        # TODO: Add multiprocessing version here
        self.transformer_list = [x for x in transformer_list if x is not None]
        self.n_jobs = n_jobs

    def _fit_singleproc(self, dataset: LAMLDataset) -> "UnionTransformer":
        """Single process version of fit.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        # TODO: just for structure. Add parallel version later
        fnames = []
        for trf in self.transformer_list:
            trf.fit(dataset)
            fnames.append(trf.features)

        self.features = fnames

        return self

    def _fit_multiproc(self, dataset: LAMLDataset) -> "UnionTransformer":
        """Multi-process version of fit.

        Args:
            dataset: Datatset to fit on.

        Returns:
            self.

        """
        raise NotImplementedError

    def fit(self, dataset: LAMLDataset) -> "UnionTransformer":
        """Fit transformers in parallel.

         Output names - concatenation of features names with no prefix.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        if self.n_jobs == 1:
            return self._fit_singleproc(dataset)
        else:
            return self._fit_multiproc(dataset)

    def _fit_transform_singleproc(self, dataset: LAMLDataset) -> List[LAMLDataset]:
        """Single process version of ``fit_transform``.

        Args:
            dataset: Dataset to transform.

        Returns:
            List of datasets with new features.

        """
        res = []
        fnames = []
        current = 0
        for n in range(len(self.transformer_list)):
            trf = self.transformer_list[n]
            ds = trf.fit_transform(dataset)
            if ds is not None:
                fnames.append(trf.features)
                res.append(ds)
                c = self.transformer_list[current]
                self.transformer_list[current] = trf
                self.transformer_list[n] = c
                current += 1
        self.transformer_list = self.transformer_list[:current]

        self.features = fnames

        return res

    def _fit_transform_multiproc(self, dataset: LAMLDataset) -> List[LAMLDataset]:
        """Multiproc version of fit_transform.

        Args:
            dataset: Dataset to fit on.

        Return:
            Now not implemented. Will be: Dataset with new features.

        """
        raise NotImplementedError

    def fit_transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Fit and transform transformers in parallel.
         Output names - concatenation of features names with no prefix.

        Args:
            dataset: Dataset to fit and transform on.

        Returns:
            Dataset with new features.

        """
        if self.n_jobs == 1:
            res = self._fit_transform_singleproc(dataset)
        else:
            res = self._fit_transform_multiproc(dataset)

        res = concatenate(res) if len(res) > 0 else None

        return res

    def _transform_singleproc(self, dataset: LAMLDataset) -> List[LAMLDataset]:
        """Single process version of transform.

        Args:
            dataset: Dataset to transform.

        Returns:
            List of dataset with new features.

        """
        res = []

        for trf in self.transformer_list:
            ds = trf.transform(dataset)
            res.append(ds)

        return res

    def _transform_multiproc(self, dataset: LAMLDataset) -> List[LAMLDataset]:
        """Multi-process version of transform.

        Args:
            dataset: Dataset to transform.

        Returns:
            List of datasets with new features.

        """
        raise NotImplementedError

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Apply transformers in parallel.
         Output names - concatenation of features names with no prefix.

        Args:
            dataset: Dataset to fit and transform on.

        Returns:
            Dataset with new features.

        """
        if self.n_jobs == 1:
            res = self._transform_singleproc(dataset)
        else:
            res = self._transform_multiproc(dataset)

        res = concatenate(res)

        return res


class ColumnsSelector(LAMLTransformer):
    """
    Select columns to pass to another transformers (or feature selection).
    """

    def __init__(self, keys: Sequence[str]):
        """

        Args:
            keys: Columns names.

        """
        self.keys = keys

    def fit(self, dataset: LAMLDataset) -> "ColumnsSelector":
        """Empty fit method - just set features.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        self.features = self.keys

        return self

    def transform(self, dataset: LAMLTransformer) -> LAMLTransformer:
        """Select given keys from dataset.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with selected features.

        """
        # to avoid coping if not needed
        if len(self.keys) == len(dataset.features) and all((x == y for (x, y) in zip(self.keys, dataset.features))):
            return dataset
        return dataset[:, self.keys]


class ColumnwiseUnion(UnionTransformer):
    # TODO: Union is not ABC !! NotImplemented - means not done right now
    """
    Apply 1 columns transformer to all columns.
    Example: encode all categories with single category encoders.
    """

    def __init__(self, transformer: LAMLTransformer, n_jobs: int = 1):
        """
        Create list of identical transformers from one.

        Args:
            transformer: Dataset - base transformer.
        """
        self.base_transformer = transformer
        self.n_jobs = n_jobs

    def _create_transformers(self, dataset: LAMLDataset):
        """
        Make a copies of base transformer.

        Args:
            dataset: Dataset with input features.

        """
        self.transformer_list = []

        for i in dataset.features:
            pipe = [ColumnsSelector([i]), deepcopy(self.base_transformer)]

            self.transformer_list.append(SequentialTransformer(pipe))

    def fit(self, dataset: LAMLDataset):
        """Create transformer list and then fit.

        Args:
            dataset: Dataset with input features.

        Returns:
            self.

        """
        self.features = dataset.features
        self._create_transformers(dataset)

        return super().fit(dataset)

    def fit_transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Create transformer list and then fit and transform.

        Args:
            dataset: Dataset with input features.

        Returns:
            Dataset with new features.

        """
        self.features = dataset.features
        self._create_transformers(dataset)

        return super().fit_transform(dataset)


class BestOfTransformers(LAMLTransformer):
    """Apply multiple transformers and select best."""

    def __init__(self, transformer_list: Sequence[LAMLTransformer], criterion: Callable):
        """Create selector from candidate list and selection criterion.

        Args:
            transformer_list: Sequence of transformers.
            criterion: Score function (greater is better).

        """
        self.transformer_list = transformer_list
        self.criterion = criterion

    def fit(self, dataset: LAMLDataset):
        """Empty method - raise error.
         This transformer supports only ``fit_transform``.

        Args:
            dataset: LAMLDataset to fit on.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("Support only fit_transform in BestOfTransformers")

    def fit_transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Fit transform all transformers and then select best.

        Args:
            dataset: Dataset to fit and then transform.

        Returns:
            Dataset with new features.

        """
        res = []

        for trf in self.transformer_list:
            ds = trf.fit_transform(dataset)
            res.append(ds)

        self.scores = np.array([self.criterion(ds) for ds in res])
        idx = self.scores.argmax()
        self.best_transformer = self.transformer_list[idx]
        self.features = self.best_transformer.features

        return res[idx]

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Make transform by the best selected transformer.

        Args:
            dataset: Dataset with input features.

        Returns:
            Dataset with new features.

        """
        return self.best_transformer.transform(dataset)


class ConvertDataset(LAMLTransformer):
    """Convert dataset to given type."""

    def __init__(self, dataset_type: ClassVar[LAMLDataset]):
        """

        Args:
            dataset_type: Type to which to convert.

        """
        self.dataset_type = dataset_type

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Dataset type should implement ``from_dataset`` method.

        Args:
            dataset: Dataset to convert.

        Returns:
            Converted dataset.

        """
        return self.dataset_type.from_dataset(dataset)


class ChangeRoles(LAMLTransformer):
    """Change data roles (include dtypes etc)."""

    def __init__(self, roles: Roles):
        """
        Args:
            roles: New roles for dataset.

        """
        self.roles = roles

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Paste new roles into dataset.

        Args:
            dataset: Dataset to transform.

        Returns:
            New dataset.

        """
        data, features, _ = dataset.data, dataset.features, dataset.roles
        dataset = dataset.empty()
        dataset.set_data(data, features, self.roles)

        return dataset
