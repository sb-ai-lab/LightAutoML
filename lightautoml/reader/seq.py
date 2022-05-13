"""Calculating inds for different TS datasets."""

import numpy as np
import pandas as pd

from ..dataset.roles import DatetimeRole


def sliding_window_view(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1,) + (window,)
    strides = a.strides[:-1] + (a.strides[-1],) + a.strides[-1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window, step=1, from_last=True):
    "from_last == True - will cut first step-1 elements"
    sliding_window = (
        sliding_window_view(a, window)
        if np.__version__ < "1.20"
        else np.lib.stride_tricks.sliding_window_view(a, window)
    )
    return sliding_window[(len(a) - window) % step if from_last else 0:][::step]


class TopInd:
    def __init__(
            self, n_target=7, history=100, step=3, from_last=True, test_last=True, roles=None, scheme=None, **kwargs
    ):
        self.n_target = n_target
        self.history = history
        self.step = step
        self.from_last = from_last
        self.test_last = test_last
        self.roles = roles

    def read(self, data, plain_data=None):
        self.len_data = len(data)
        self.date_col = [col for col, role in self.roles.items() if isinstance(role, DatetimeRole)][0]
        self.time_delta = pd.to_datetime(data[self.date_col]).diff().iloc[1]

        ## TO DO:
        # add asserts

    def create_test(self, data=None, plain_data=None):
        # for predicting future
        return rolling_window(
            np.arange(self.len_data if data is None else len(data)), self.history, self.step, self.from_last
        )[-1 if self.test_last else 0:, :]

    def _create_test(self, data=None, plain_data=None):
        # for predicting future
        return rolling_window(
            np.arange(self.len_data if data is None else len(data)), self.history, self.step, self.from_last
        )[-1 if self.test_last else 0:, :]

    def _create_data(self, data=None, plain_data=None):

        return rolling_window(
            np.arange(self.len_data if data is None else len(data))[: -self.n_target],
            self.history,
            self.step,
            self.from_last,
        )

    def _create_target(self, data=None, plain_data=None):
        return rolling_window(
            np.arange(self.len_data if data is None else len(data))[self.history:],
            self.n_target,
            self.step,
            self.from_last,
        )

    def _get_ids(self, data=None, plain_data=None, func=None, cond=None):
        date_col = pd.to_datetime(data[self.date_col])
        vals = pd.to_datetime(data[self.date_col]).diff().fillna(self.time_delta).values
        ids = list(np.argwhere(vals != self.time_delta).flatten())
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

    def create_data(self, data=None, plain_data=None):
        return self._get_ids(data, plain_data, self._create_data, self.n_target + self.history)

    def create_test(self, data=None, plain_data=None):
        return self._get_ids(data, plain_data, self._create_test, self.history)

    def create_target(self, data=None, plain_data=None):
        return self._get_ids(data, plain_data, self._create_target, self.n_target + self.history)


class IDSInd:
    def __init__(self, scheme, **kwargs):
        self.scheme = scheme

    def read(self, data, plain_data=None):
        self.len_data = len(data)

        ## TO DO:
        # add asserts

    def create_test(self, data, plain_data):
        # for predicting future
        return self.create_data(data, plain_data)

    def create_data(self, data, plain_data):
        ids = data.reset_index().groupby(self.scheme["from_id"])["index"].apply(self._to_list).to_dict()
        s = plain_data[self.scheme["to_id"]].map(ids)
        s.loc[s.isna()] = [[] for i in range(len(s.loc[s.isna()]))]
        return s.values

    def create_target(self, data=None, plain_data=None):
        return None

    @staticmethod
    def _to_list(x):
        return list(x)
