"""Tabular data utils."""

import os
import warnings

from copy import copy
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from pandas import DataFrame
from sqlalchemy import create_engine


def get_filelen(fname: str) -> int:
    """Get length of csv file.

    Args:
        fname: File name.

    Returns:
        Length of file.

    """
    cnt_lines = -1
    with open(fname, "rb") as fin:
        for line in fin:
            if len(line.strip()) > 0:
                cnt_lines += 1
    return cnt_lines


def get_batch_ids(arr, batch_size):
    """Generator of batched sequences.

    Args:
        arr: Sequense.
        batch_size: Batch size.

    Returns:
        Generator.

    """
    n = 0
    while n < len(arr):
        yield arr[n : n + batch_size]
        n += batch_size


def get_file_offsets(
    file: str, n_jobs: Optional[int] = None, batch_size: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """

    Args:
        file: File path.
        n_jobs: Number of jobs for multiprocessing.
        batch_size: Batch size.

    Returns:
        Offsets tuple.

    """
    assert n_jobs is not None or batch_size is not None, "One of n_jobs or batch size should be defined"

    lens = []
    with open(file, "rb") as f:
        # skip header
        header_len = len(f.readline())
        # get row lens
        length = 0
        for row in f:
            if len(row.strip()) > 0:
                lens.append(length)
                length += len(row)

    lens = np.array(lens, dtype=np.int64) + header_len

    if batch_size:
        indexes = list(get_batch_ids(lens, batch_size))
    else:
        indexes = np.array_split(lens, n_jobs)

    offsets = [x[0] for x in indexes]
    cnts = [x.shape[0] for x in indexes]

    return offsets, cnts


def _check_csv_params(**read_csv_params: dict):
    """

    Args:
        **read_csv_params: Read parameters.

    Returns:
        New parameters.

    """
    for par in ["skiprows", "nrows", "index_col", "header", "names", "chunksize"]:
        if par in read_csv_params:
            read_csv_params.pop(par)
            warnings.warn(
                "Parameter {0} will be ignored in parallel mode".format(par),
                UserWarning,
            )

    return read_csv_params


def read_csv_batch(file: str, offset, cnt, **read_csv_params):
    """Read batch of data from csv.

    Args:
        file: File path.
        offset: Start of file.
        cnt: Number of rows to read.
        **read_csv_params: Handler parameters.

    Returns:
        Read data.

    """
    read_csv_params = copy(read_csv_params)
    if read_csv_params is None:
        read_csv_params = {}

    try:
        usecols = read_csv_params.pop("usecols")
    except KeyError:
        usecols = None

    header = pd.read_csv(file, nrows=0, **read_csv_params).columns

    with open(file, "rb") as f:
        f.seek(offset)
        data = pd.read_csv(f, header=None, names=header, chunksize=None, nrows=cnt, usecols=usecols, **read_csv_params)

    return data


def read_csv(file: str, n_jobs: int = 1, **read_csv_params) -> DataFrame:
    """Read data from csv.

    Args:
        file: File path.
        n_jobs: Number of workers.
        **read_csv_params: Handler parameters.

    Returns:
        Read data.
    """
    if n_jobs == 1:
        return pd.read_csv(file, **read_csv_params)

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    _check_csv_params(**read_csv_params)
    offsets, cnts = get_file_offsets(file, n_jobs)

    with Parallel(n_jobs) as p:
        res = p(
            delayed(read_csv_batch)(file, offset=offset, cnt=cnt, **read_csv_params)
            for (offset, cnt) in zip(offsets, cnts)
        )

    res = pd.concat(res, ignore_index=True)

    return res


class Batch:
    """
    Class to wraps batch of data in different formats.
    Default - batch of DataFrame.
    """

    @property
    def data(self) -> DataFrame:
        """Get data from Batch object.

        Returns:
            Data.

        """
        return self._data

    def __init__(self, data):
        self._data = data


class FileBatch(Batch):
    """
    Batch of csv file.
    """

    @property
    def data(self) -> DataFrame:
        """Get data from Batch object.

        Returns:
            Read data.

        """
        data_part = read_csv_batch(self.file, cnt=self.cnt, offset=self.offset, **self.read_csv_params)

        return data_part

    def __init__(self, file, offset, cnt, read_csv_params):
        """
        Args:
            file: File path.
            offset: File start.
            cnt: Number of rows to read.
            read_csv_params: Additional params to :func:`pandas.read_csv`.

        """
        self.file = file
        self.offset = offset
        self.cnt = cnt
        self.read_csv_params = read_csv_params


class BatchGenerator:
    """
    Abstract - generator of batches from data.
    """

    def __init__(self, batch_size, n_jobs):
        """

        Args:
            n_jobs: Number of processes to handle.
            batch_size: Batch size. Default is ``None``, split by `n_jobs`.

        """
        if n_jobs == -1:
            n_jobs = os.cpu_count()

        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def __getitem__(self, idx) -> Batch:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DfBatchGenerator(BatchGenerator):
    """
    Batch generator from :class:`~pandas.DataFrame`.
    """

    def __init__(self, data: DataFrame, n_jobs: int = 1, batch_size: Optional[int] = None):
        """

        Args:
            data: Data used for generator.
            n_jobs: Number of processes to handle.
            batch_size: Batch size. Default is ``None``, split by `n_jobs`.

        """
        super().__init__(batch_size, n_jobs)

        self.data = data

        if self.batch_size is not None:
            self.idxs = list(get_batch_ids(np.arange(data.shape[0]), batch_size))
        else:
            self.idxs = [x for x in np.array_split(np.arange(data.shape[0]), n_jobs) if len(x) > 0]

    def __len__(self) -> int:

        if self.batch_size is not None:
            return int(np.ceil(self.data.shape[0] / self.batch_size))

        return int(self.n_jobs)

    def __getitem__(self, idx):

        return Batch(self.data.iloc[self.idxs[idx]])


class FileBatchGenerator(BatchGenerator):
    """
    Generator of batches from file.
    """

    def __init__(
        self,
        file,
        n_jobs: int = 1,
        batch_size: Optional[int] = None,
        read_csv_params: dict = None,
    ):
        """

        Args:
            file: File path.
            n_jobs: Number of processes to handle.
            batch_size: Batch size. Default is ``None``, split by `n_jobs`.
            read_csv_params: Params of reading csv file.
              Look for :func:`pandas.read_csv` params.

        """
        super().__init__(batch_size, n_jobs)

        self.file = file
        self.offsets, self.cnts = get_file_offsets(file, n_jobs, batch_size)

        if read_csv_params is None:
            read_csv_params = {}

        self.read_csv_params = read_csv_params

    def __len__(self) -> int:
        return len(self.cnts)

    def __getitem__(self, idx):
        return FileBatch(self.file, self.offsets[idx], self.cnts[idx], self.read_csv_params)


class SqlDataSource:
    def __init__(
        self,
        connection_string: str,
        query: str,
        index: Optional[Union[str, List[str]]] = None,
    ):
        """

        Data wrapper for SQL connection

        Args:
            connection_string: database url; for reference see
            https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
            query: SQL query to obtain data from
            index: optional index column to be removed from the query result; can be None, str of List[str]
        """
        self.engine = create_engine(connection_string)
        self.query = query
        self.index = index
        self._data = None

    @property
    def data(self):
        """
        Get data associated with the query as :class:`~pandas.DataFrame`

        Returns:
            :class:`~pandas.DataFrame`
        """
        if self._data is None:
            with self.engine.begin() as conn:
                self._data = pd.read_sql(self.query, conn, index_col=self.index)
        return self._data

    def get_batch_generator(self, n_jobs: int = 1, batch_size: int = None):
        """
        Access data with batch generator
        Args:
            n_jobs: Number of processes to read file.
            batch_size: Number of entries in one batch.

        Returns:
            DfBatchGenerator object
        """
        return DfBatchGenerator(self.data, n_jobs, batch_size)


ReadableToDf = Union[str, np.ndarray, DataFrame, Dict[str, np.ndarray], Batch]


def read_data(
    data: ReadableToDf,
    features_names: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
    read_csv_params: Optional[dict] = None,
) -> Tuple[DataFrame, Optional[dict]]:
    """Get :class:`~pandas.DataFrame` from different data formats.

    Note:
        Supported now data formats:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
              For example, ``{'data': X...}``. In this case,
              roles are optional, but `train_features`
              and `valid_features` required.
            - :class:`pandas.DataFrame`.

    Args:
        data: Readable to DataFrame data.
        features_names: Optional features names if ``numpy.ndarray``.
        n_jobs: Number of processes to read file.
        read_csv_params: Params to read csv file.

    Returns:
        Tuple with read data and new roles mapping.

    """
    if read_csv_params is None:
        read_csv_params = {}
    # case - new process
    if isinstance(data, Batch):
        return data.data, None

    if isinstance(data, DataFrame):
        return data, None
    # case - single array passed to inference
    if isinstance(data, np.ndarray):
        return DataFrame(data, columns=features_names), None

    # case - dict of array args passed
    if isinstance(data, dict):
        df = DataFrame(data["data"], columns=features_names)
        upd_roles = {}
        for k in data:
            if k != "data":
                name = "__{0}__".format(k.upper())
                assert name not in df.columns, "Not supported feature name {0}".format(name)
                df[name] = data[k]
                upd_roles[k] = name
        return df, upd_roles

    if isinstance(data, str):
        if data.endswith(".feather"):
            # TODO: check about feather columns arg
            data = pd.read_feather(data)
            if read_csv_params["usecols"] is not None:
                data = data[read_csv_params["usecols"]]
            return data, None

        if data.endswith(".parquet"):
            return pd.read_parquet(data, columns=read_csv_params["usecols"]), None

        else:
            return read_csv(data, n_jobs, **read_csv_params), None

    if isinstance(data, SqlDataSource):
        return data.data, None

    raise ValueError("Input data format is not supported")


def read_batch(
    data: ReadableToDf,
    features_names: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
    batch_size: Optional[int] = None,
    read_csv_params: Optional[dict] = None,
) -> Iterable[BatchGenerator]:
    """Read data for inference by batches for simple tabular data

    Note:
        Supported now data formats:

            - Path to ``.csv``, ``.parquet``, ``.feather`` files.
            - :class:`~numpy.ndarray`, or dict of :class:`~numpy.ndarray`.
              For example, ``{'data': X...}``. In this case,
              roles are optional, but `train_features`
              and `valid_features` required.
            - :class:`pandas.DataFrame`.

    Args:
        data: Readable to DataFrame data.
        features_names: Optional features names if ``numpy.ndarray``.
        n_jobs: Number of processes to read file.
        read_csv_params: Params to read csv file.

    Returns:
        Generator of batches.

    """
    if read_csv_params is None:
        read_csv_params = {}

    if isinstance(data, DataFrame):
        return DfBatchGenerator(data, n_jobs=n_jobs, batch_size=batch_size)

    # case - single array passed to inference
    if isinstance(data, np.ndarray):
        return DfBatchGenerator(
            DataFrame(data, columns=features_names),
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

    if isinstance(data, str):
        if not (data.endswith(".feather") or data.endswith(".parquet")):
            return FileBatchGenerator(
                data, n_jobs, batch_size, read_csv_params
            )  # read_csv(data, n_jobs, **read_csv_params)

        else:
            data, _ = read_data(data, features_names, n_jobs, read_csv_params)
            return DfBatchGenerator(data, n_jobs=n_jobs, batch_size=batch_size)

    if isinstance(data, SqlDataSource):
        return data.get_batch_generator(n_jobs, batch_size)

    raise ValueError("Data type not supported")
