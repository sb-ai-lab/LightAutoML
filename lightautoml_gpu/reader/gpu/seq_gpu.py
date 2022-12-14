"""Calculating inds for different TS datasets (GPU version)."""

import numpy as np
import pandas as pd

import cudf
import dask_cudf

from lightautoml_gpu.dataset.roles import DatetimeRole
from ..seq import TopInd, IDSInd


class TopIndGPU(TopInd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_cpu(self):
        n_target = self.n_target
        history = self.history
        step = self.step
        from_last = self.from_last
        test_last = self.test_last
        roles = self.roles
        self.__class__ = TopInd
        self.n_target = n_target
        self.history = history
        self.step = step
        self.from_last = from_last
        self.test_last = test_last
        self.roles = roles
        return self

    @staticmethod
    def _timedelta(x):
        delta = cudf.to_datetime(x).diff().iloc[-1]
        if delta <= pd.Timedelta(days=1):
            return cudf.to_datetime(x).diff().fillna(delta).values_host, delta

        if delta > pd.Timedelta(days=360):
            d = cudf.to_datetime(x).dt.year.diff()
            delta = d.iloc[-1]
            return d.fillna(delta).values_host, delta
        elif delta > pd.Timedelta(days=27):
            d = cudf.to_datetime(x).dt.month.diff() + 12 * cudf.to_datetime(x).dt.year.diff()
            delta = d.iloc[-1]
            return d.fillna(delta).values_host, delta
        else:
            return cudf.to_datetime(x).diff().fillna(delta).values_host, delta

    def read(self, data, plain_data=None):
        self.len_data = len(data)
        self.date_col = [col for col, role in self.roles.items() if isinstance(role, DatetimeRole)][0]
        time_data = None
        if isinstance(data, cudf.DataFrame):
            time_data = data[self.date_col]
        elif isinstance(data, dask_cudf.DataFrame):
            time_data = data[self.date_col].compute()
        else:
            raise TypeError("wrong data type for read() in "
                            + self.__class__.__name__)

        self.time_delta = self._timedelta(time_data)[1]
        return self
        # TODO: add asserts

    def _get_ids(self, data=None, plain_data=None, func=None, cond=None):
        time_data = None
        if isinstance(data, cudf.DataFrame):
            time_data = data[self.date_col]
        elif isinstance(data, dask_cudf.DataFrame):
            time_data = data[self.date_col].compute()
        else:
            raise TypeError("wrong data type for read() in "
                            + self.__class__.__name__)
        date_col = cudf.to_datetime(time_data).to_pandas()
        vals, time_delta = self._timedelta(time_data)
        ids = list(np.argwhere(vals != time_delta).flatten())
        prev = 0
        inds = []
        for split in ids + [len(date_col)]:
            segment = date_col.iloc[prev:split]
            if len(segment) > cond:
                ind = func(segment) + prev
                inds.append(ind)
            prev = split
        inds = np.vstack(inds)
        return inds


class IDSIndGPU(IDSInd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_cpu(self):
        scheme = self.scheme
        self.__class__ = IDSInd
        self.scheme = scheme
        return self

    def create_data(self, data, plain_data):
        if isinstance(data, cudf.DataFrame):
            data = data[self.scheme["from_id"]]
        elif isinstance(data, dask_cudf.DataFrame):
            data = data[self.scheme["from_id"]].compute()
        else:
            raise TypeError("wrong data type for read() in "
                            + self.__class__.__name__)
        ids = data.reset_index().groupby(self.scheme["from_id"])["index"].agg('collect').to_pandas().to_dict()

        if isinstance(plain_data, cudf.DataFrame):
            plain_data = plain_data[self.scheme["to_id"]].to_pandas()
        elif isinstance(plain_data, dask_cudf.DataFrame):
            plain_data = plain_data[self.scheme["to_id"]].compute().to_pandas()
        else:
            raise TypeError("wrong data type for read() in "
                            + self.__class__.__name__)
        s = plain_data.map(ids)
        s.loc[s.isna()] = [[] for i in range(len(s.loc[s.isna()]))]
        result = np.array([x for x in s.values])
        return result
