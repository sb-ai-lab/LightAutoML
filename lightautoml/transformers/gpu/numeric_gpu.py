"""Numeric features transformers."""

from typing import Union

import cudf
import cupy as cp
import numpy as np
import dask.array as da

from lightautoml.dataset.gpu.gpu_dataset import CudfDataset, CupyDataset, DaskCudfDataset
from lightautoml.dataset.np_pd_dataset import NumpyDataset, PandasDataset
from lightautoml.dataset.roles import CategoryRole, NumericRole
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.numeric import numeric_check

# type - something that can be converted to pandas dataset
CupyTransformable = Union[
    NumpyDataset, PandasDataset, CupyDataset, CudfDataset, DaskCudfDataset
]
GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class NaNFlagsGPU(LAMLTransformer):
    """Create NaN flags (GPU version)."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "nanflg_gpu"

    def __init__(self, nan_rate: float = 0.005):
        """

        Args:
            nan_rate: Nan rate cutoff.

        """
        self.nan_rate = nan_rate

    def _fit_cupy(self, dataset: CupyTransformable):

        dataset = dataset.to_cupy()
        data = dataset.data
        # fit ...
        ds_nan_rate = cp.isnan(data).mean(axis=0)
        self.nan_cols = [
            name
            for (name, nan_rate) in zip(dataset.features, ds_nan_rate)
            if nan_rate > self.nan_rate
        ]
        self._features = list(self.nan_cols)

        return self

    def _fit_daskcudf(self, dataset: DaskCudfDataset):

        ds_nan_rate = dataset.data.isna().mean().compute().values
        self.nan_cols = [
            name
            for (name, nan_rate) in zip(dataset.features, ds_nan_rate)
            if nan_rate > self.nan_rate
        ]
        self._features = list(self.nan_cols)

        return self

    def fit(self, dataset: GpuDataset):
        """Extract nan flags (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)

        if isinstance(dataset, DaskCudfDataset):
            self._fit_daskcudf(dataset)
        else:
            self._fit_cupy(dataset)

        return self

    def _transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:

        dataset = dataset.to_cupy()
        output = dataset.empty()

        if len(self.nan_cols) > 0:
            nans = dataset[:, self.nan_cols].data
            # transform
            # new_arr = nans.isna().astype(cp.float32)
            new_arr = cp.isnan(nans).astype(cp.float32)

            # create resulted
            output.set_data(new_arr, self.features, NumericRole(np.float32))
        else:
            return None

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        output = dataset.empty()

        if len(self.nan_cols) > 0:
            data = dataset.data[self.nan_cols].isna().astype(np.float32)

            output.set_data(data, self.features, NumericRole(np.float32))

        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform - extract null flags (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            Cupy or DaskCudf dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class FillnaMedianGPU(LAMLTransformer):
    """Fillna with median (GPU version)."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillnamed_gpu"

    def _fit_cupy(self, dataset: CupyTransformable):
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        self.meds = cp.nanmedian(data, axis=0)
        self.meds[cp.isnan(self.meds)] = 0

        return self

    def _fit_daskcudf(self, dataset: DaskCudfDataset):

        self.meds = da.nanmedian(dataset.data.to_dask_array(lengths=True), axis=0)
        self.meds = cudf.Series(self.meds.compute(),index=dataset.data.columns).astype(cp.float32).fillna(0.0)

        return self

    def fit(self, dataset: CupyTransformable):
        """Estimate medians (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            self._fit_daskcudf(dataset)
        else:
            self._fit_cupy(dataset)

        return self

    def _transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
        # transform
        data = cp.where(cp.isnan(data), self.meds, data)
        # data = cudf.DataFrame(data, index=data.index, columns=self.features)

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        data = dataset.data

        data = data.fillna(self.meds)
        output = dataset.empty()
        
        output.set_data(data, self.features, NumericRole(np.float32))
        return output

    def transform(self, dataset: CupyTransformable) -> GpuDataset:
        """Transform - fillna with medians (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            Cupy or DaskCudf dataset with encoded labels.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class FillInfGPU(LAMLTransformer):
    """Fill inf with nan to handle as nan value."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "fillinf_gpu"

    def _inf_to_nan(self, data: cudf.DataFrame) -> cudf.DataFrame:
        output = cp.where(
            cp.isinf(data.fillna(cp.nan).values), cp.nan, data.fillna(cp.nan).values
        )
        return cudf.DataFrame(output, columns=self.features, index=data.index)

    def _transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        # transform
        data = cp.where(cp.isinf(data), cp.nan, data)

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        data = dataset.data.map_partitions(
            self._inf_to_nan, meta=cudf.DataFrame(columns=self.features)
        ).persist()

        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output

    def transform(self, dataset: CupyTransformable) -> GpuDataset:
        """Replace inf to nan (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            Cupy or DaskCudf dataset with encoded labels.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class LogOddsGPU(LAMLTransformer):
    """Convert probs to logodds (GPU version)."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "logodds_gpu"

    def _transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:

        dataset = dataset.to_cupy()
        data = dataset.data
        data = cp.clip(data, 1e-7, 1 - 1e-7)

        data = cp.log(data / (1 - data))

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        out = dataset.data.persist().clip(1e-7, 1.0 - 1e-7)
        out = (1.0 * out / (1.0 - out)).map_partitions(np.log)

        out = out.rename(columns=dict(zip(dataset.features, self.features)))

        output = dataset.empty()
        output.set_data(out, self.features, NumericRole(np.float32))

        return output

    def transform(self, dataset: CupyTransformable) -> GpuDataset:
        """Transform - convert num values to logodds.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            Cupy or DaskCudf dataset with encoded labels.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class StandardScalerGPU(LAMLTransformer):
    """Classic StandardScaler (GPU version)."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "scaler_gpu"

    def _fit_cupy(self, dataset: CupyTransformable):

        dataset = dataset.to_cupy()
        data = dataset.data

        self.means = cp.nanmean(data, axis=0)
        self.stds = cp.nanstd(data, axis=0)
        # Fix zero stds to 1
        self.stds[(self.stds == 0) | cp.isnan(self.stds)] = 1

        return self

    def _fit_daskcudf(self, dataset: DaskCudfDataset):

        self.means = dataset.data.mean(skipna=True).compute().values
        self.stds = dataset.data.std(skipna=True).compute().values
        self.stds[(self.stds == 0) | cp.isnan(self.stds)] = 1
        return self

    def fit(self, dataset: CupyTransformable):
        """Estimate means and stds (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._fit_daskcudf(dataset)

        else:
            return self._fit_cupy(dataset)

        return self

    def _standardize(self, data):
        output = (data.values - self.means) / self.stds
        return cudf.DataFrame(output, columns=self.features, index=data.index)

    def _transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        # transform
        data = (data - self.means) / self.stds

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(np.float32))

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        data = dataset.data.map_partitions(
            self._standardize, meta=cudf.DataFrame(columns=self.features)
        ).persist()
        output = dataset.empty()
        output.set_data(data, self.features, NumericRole(np.float32))
        return output

    def transform(self, dataset: CupyTransformable) -> GpuDataset:
        """Scale test data (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of numeric features.

        Returns:
            Cupy or DaskCudf dataset with encoded labels.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)


class QuantileBinningGPU(LAMLTransformer):
    """Discretization of numeric features by quantiles (GPU version)."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "qntl_gpu"

    def __init__(self, nbins: int = 10):
        """

        Args:
            nbins: maximum number of bins.

        """
        self.nbins = nbins

    def _fit_cupy(self, dataset: CupyTransformable):
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
        sl = cp.isnan(data)
        grid = cp.linspace(0, 1, self.nbins + 1)[1:-1]
        self.bins = []
        for n in range(data.shape[1]):
            q = cp.quantile(data[:, n][~sl[:, n]], q=grid)
            q = cp.unique(q)
            self.bins.append(q)
        return self

    def _fit_daskcudf(self, dataset: DaskCudfDataset):

        data = dataset.data
        grid = np.linspace(0, 1, self.nbins + 1)[1:-1]
        self.bins = []
        for col in data.columns:
            q = data[col].dropna().quantile(grid).persist()
            q = q.unique().astype(cp.float32)
            self.bins.append(q.compute().values)
        return self

    def fit(self, dataset: CupyTransformable):
        """Estimate bins borders (GPU version).

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of numeric features.

        Returns:
            self.

        """

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._fit_daskcudf(dataset)
        else:
            return self._fit_cupy(dataset)

    def _digitize(self, data):
        cp_data = data.values
        sl = cp.isnan(cp_data)
        inds = data.index
        data = cp.zeros(cp_data.shape, dtype=cp.int32)
        for n, b in enumerate(self.bins):
            data[:, n] = (
                cp.searchsorted(b, cp.where(sl[:, n], cp.inf, cp_data[:, n])) + 1
            )

        data = cp.where(sl, 0, data)
        output = cudf.DataFrame(data, columns=self.features, index=inds)
        return output

    def _transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:

        dataset = dataset.to_cupy()
        data = dataset.data

        # transform
        sl = cp.isnan(data)

        new_data = cp.zeros(data.shape, dtype=cp.int32)

        for n, b in enumerate(self.bins):
            new_data[:, n] = (
                cp.searchsorted(b, cp.where(sl[:, n], cp.inf, data[:, n])) + 1
            )

        new_data = cp.where(sl, 0, new_data)

        # create resulted
        output = dataset.empty()
        output.set_data(
            new_data, self.features, CategoryRole(np.int32, label_encoded=True)
        )

        return output

    def _transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:

        new_data = dataset.data.map_partitions(
            self._digitize, meta=cudf.DataFrame(columns=self.features)
        ).persist()
        output = dataset.empty()
        output.set_data(
            new_data, self.features, CategoryRole(np.int32, label_encoded=True)
        )
        return output

    def transform(self, dataset: CupyTransformable) -> GpuDataset:
        """Apply bin borders.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of numeric features.

        Returns:
            Cupy or DaskCudf dataset with encoded labels.

        """

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self._transform_daskcudf(dataset)
        else:
            return self._transform_cupy(dataset)
