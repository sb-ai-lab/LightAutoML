import numpy as np
import pandas as pd
from numba import jit
import string

from lightautoml_gpu.reader.gpu.seq_reader_gpu import DictToCudfSeqReader
from lightautoml_gpu.reader.gpu.seq_reader_gpu import DictToDaskCudfSeqReader
from lightautoml_gpu.reader.base import DictToPandasSeqReader
from lightautoml_gpu.tasks import Task

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))


@jit(nopython=True)
def gen_cols(n_cols):
    cols = [""] * n_cols
    for i in range(n_cols):
        cols[i] = "col_" + str(i)
    return cols


def gen_string_data(n, n_str):
    string_db = ["algorithm", "analog", "app", "application", "array",
                 "backup", "bandwidth", "binary", "bit", "byte"]
    inds = np.random.randint(0, len(string_db), (n, n_str))
    output = np.empty(inds.shape, dtype=object)
    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            output[i][j] = string_db[inds[i][j]]

    return output


def gen_data_single_target(n: int, n_num: int, n_cat: int, n_date: int,
                           n_str: str, max_n_cat: int, n_ids: int,
                           max_ids: list = None, cols: list = None):
    n_cols = n_num + n_cat + n_str + n_date + n_ids
    cols = gen_cols(n_cols) if cols is None else cols
    data = np.random.random((n, n_num)) * 100 - 50

    category_data = np.random.randint(0, np.random.randint(1, max_n_cat), (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000, (n, n_date))\
        .astype(np.dtype("timedelta64[D]")) \
        + np.datetime64("2018-01-01")

    if max_ids is None:
        id_data = np.arange(n, dtype=int)[:, np.newaxis]\
            * np.ones(n_ids, dtype=int)[:, np.newaxis].T
        for elem in id_data.T:
            np.random.shuffle(elem)
    else:
        id_data = np.array(np.random.random((n, n_ids)) * max_ids // 1,
                           dtype=int)

    data = pd.DataFrame(data, columns=cols[:n_num]).astype('f')

    # ix = [(row, col) for row in range(data.shape[0]) \
    #                 for col in range(data.shape[1])]
    # for row, col in random.sample(ix, int(round(.1*len(ix)))):
    #    data.iat[row, col] = np.nan

    nn = len(data.columns)
    for i in range(n_cat):
        data[cols[nn + i]] = pd.Series(category_data[:, i]).astype('f')
    nn = len(data.columns)
    for i in range(n_str):
        data[cols[nn + i]] = pd.Series(string_data[:, i]).astype(object)
    nn = len(data.columns)
    for i in range(n_date):
        data[cols[nn + i]] = pd.Series(date_data[:, i])
    nn = len(data.columns)
    for i in range(n_ids):
        data[cols[nn + i]] = pd.Series(id_data[:, i])

    data['TARGET'] = pd.Series(np.random.randint(0, 5, n)).astype('i')

    return 'TARGET', cols, data


if __name__ == "__main__":

    # Data preparation
    n, n_num, n_cat, n_date, n_str = 50, 3, 2, 2, 1
    max_n_cat, n_ids = 10, 1
    cols_data1 = ["a", "b", "c", "d", "e", "str1",
                  "date1", "date2", "data1_id"]
    _, _, data1 = gen_data_single_target(n, n_num, n_cat, n_date,
                                         n_str, max_n_cat, n_ids,
                                         cols=cols_data1)

    n, n_num, n_cat, n_date, n_str = 350, 2, 2, 0, 0
    max_n_cat, n_ids = 5, 1
    cols_data2 = ["h", "i", "j", "k", "data2_id"]
    _, _, data2 = gen_data_single_target(n, n_num, n_cat, n_date,
                                         n_str, max_n_cat, n_ids,
                                         cols=cols_data2)

    max_ids = [50, 100]
    n, n_num, n_cat, n_date = 1000, 4, 6, 2
    n_str, max_n_cat, n_ids = 2, 15, 2
    target, cols, train = gen_data_single_target(n, n_num, n_cat,
                                                 n_date, n_str,
                                                 max_n_cat, n_ids,
                                                 max_ids)

    n = 200
    _, _, test = gen_data_single_target(n, n_num, n_cat, n_date,
                                        n_str, max_n_cat, n_ids,
                                        max_ids)
    seq_params = {'data1': {'case': 'ids',
                            'params': {},
                            'scheme': {'to': 'plain',
                                       'from_id': 'data1_id',
                                       'to_id': 'col_14'},
                            },
                  'data2': {'case': 'ids',
                            'params': {},
                            'scheme': {'to': 'plain',
                                       'from_id': 'data2_id',
                                       'to_id': 'col_15'},
                            },
                  }
    seq_data = {'data1': data1[cols_data1],
                'data2': data2[cols_data2]
                }
    X_train = {'plain': train,
               'seq': seq_data
               }
    X_test = {'plain': test,
              'seq': seq_data
              }

    # Data preparation finished

    # cpu seq reader
    task = Task('reg', metric='mae')
    roles = {'target': target}

    reader = DictToPandasSeqReader(task=task, seq_params=seq_params)
    res = reader.fit_read(X_train, roles=roles)
    res2 = reader.read(X_test)

    # cudf seq reader
    task = Task('reg', metric='mae', device='gpu')

    reader_gpu = DictToCudfSeqReader(task=task, seq_params=seq_params, n_jobs=1)
    res_gpu = reader_gpu.fit_read(X_train, roles=roles)
    res2_gpu = reader_gpu.read(X_test)

    # dask_cudf seq reader
    task = Task('reg', metric='mae', device='mgpu')

    reader_mgpu = DictToDaskCudfSeqReader(task=task, seq_params=seq_params,
                                          n_jobs=1, npartitions=2)
    res_mgpu = reader_mgpu.fit_read(X_train, roles=roles)
    res2_mgpu = reader_mgpu.read(X_test)

    inds = [1, 6, 4]
    cols = ['b', 'c']
    # slicing assert
    assert np.allclose(res2.seq_data['data1'][inds, cols].data,
                       res2_gpu.seq_data['data1'][inds, cols].data.values_host)
    assert np.allclose(res2.seq_data['data1'][inds, cols].data,
                       res2_mgpu.seq_data['data1'][inds, cols].data.compute().values_host)
    # reader output asserts
    assert res.data.shape == res_gpu.data.shape == (res_mgpu.data.shape[0].compute(), res_mgpu.data.shape[1])
    pd.testing.assert_frame_equal(res_gpu.data.to_pandas(), res.data)
    pd.testing.assert_frame_equal(res_gpu.seq_data['data2'].data.to_pandas(), res.seq_data['data2'].data)

    # to_sequence() asserts
    assert np.allclose(res2.seq_data['data2'].to_sequence([0]).data,
           res2_gpu.seq_data['data2'].get_first_frame([0]).data.values_host)
    assert np.allclose(res2.seq_data['data2'].to_sequence([0]).data,
           res2_mgpu.seq_data['data2'].get_first_frame([0]).data.compute().values_host)
    assert res2.seq_data['data2'].to_sequence().shape ==\
           res2_gpu.seq_data['data2'].get_first_frame().shape ==\
           res2_mgpu.seq_data['data2'].get_first_frame().shape
