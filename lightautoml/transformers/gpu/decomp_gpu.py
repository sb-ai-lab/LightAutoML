"""Dimension reduction transformers."""

from typing import List
from typing import Optional
from typing import Union

import numpy as np
import cupy as cp

from cuml import PCA
from cuml import TruncatedSVD

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
from lightautoml.dataset.gpu.gpu_dataset import CupyDataset
from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.transformers.base import LAMLTransformer

from lightautoml.transformers.decomposition import numeric_check

# type - something that can be converted to pandas dataset
CupyTransformable = Union[CupyDataset, CudfDataset]
CupyCSR = Union[CupyDataset, CSRSparseDataset]
GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class PCATransformer_gpu(LAMLTransformer):
    """PCA."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'pca_gpu'

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self, subs: Optional[int] = None, random_state: int = 42, n_components: int = 500):
        """

        Args:
            subs: Subsample to fit algorithm. If None - full data.
            random_state: Random state to take subsample.
            n_components: Number of PCA components

        """
        self.subs = subs
        self.random_state = random_state
        self.n_components = n_components
        self._pca = PCA
        self.pca = None

    def fit(self, dataset: CupyTransformable):
        """Fit algorithm on dataset.

        Args:
            dataset: Sparse or Cupy dataset of text features.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
        self.n_components = int(np.minimum(self.n_components, data.shape[1] - 1))
        self.pca = self._pca(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(data)

        orig_name = dataset.features[0].split('__')[-1]

        feats = np.char.array([self._fname_prefix + '_']) + np.arange(self.n_components).astype(str) + np.char.array(
            ['__' + orig_name])

        self._features = list(feats)
        return self

    def transform(self, dataset: CupyTransformable) -> CupyDataset:
        """Transform input dataset to PCA representation.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Cupy dataset with text embeddings.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
        # transform
        data = self.pca.transform(data)

        # create resulted
        output = dataset.empty().to_cupy()
        output.set_data(data, self.features, NumericRole(cp.float32))
        return output


class SVDTransformer_gpu(LAMLTransformer):
    """TruncatedSVD."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'svd_gpu'

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self, subs: Optional[int] = None, random_state: int = 42, n_components: int = 100):
        """

        Args:
            subs: Subsample to fit algorithm. If None - full data.
            random_state: Random state to take subsample.
            n_components: Number of SVD components.

        """
        self.subs = subs
        self.random_state = random_state
        self.n_components = n_components
        self._svd = TruncatedSVD
        self.svd = None

    def fit(self, dataset: CupyCSR):
        """Fit algorithm on dataset.

        Args:
            dataset: Sparse or Numpy dataset of text features.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        data = dataset.data
        self.n_components = int(np.minimum(self.n_components, data.shape[1] - 1))
        self.svd = self._svd(n_components=self.n_components, random_state=self.random_state)
        self.svd.fit(data)

        orig_name = dataset.features[0].split('__')[-1]

        feats = np.char.array([self._fname_prefix + '_']) + np.arange(self.n_components).astype(str) + np.char.array(
            ['__' + orig_name])

        self._features = list(feats)
        return self

    def transform(self, dataset: CupyCSR) -> CupyDataset:
        """Transform input dataset to SVD representation.

        Args:
            dataset: Sparse or Cupy dataset of text features.

        Returns:
            Cupy dataset with text embeddings.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        data = dataset.data
        # transform
        data = self.svd.transform(data)

        # create resulted
        output = dataset.empty().to_cupy()
        output.set_data(data, self.features, NumericRole(np.float32))
        return output
