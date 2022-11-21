from time import perf_counter
import numpy as np
import pandas as pd
import random
from numba import jit
import string

from lightautoml.reader.gpu.seq_reader_gpu import DictToCudfSeqReader
from lightautoml.reader.gpu.seq_reader_gpu import DictToDaskCudfSeqReader
from lightautoml.reader.base import DictToPandasSeqReader
from lightautoml.tasks import Task

from lightautoml.transformers.seq import SeqNumCountsTransformer
from lightautoml.transformers.seq import SeqStatisticsTransformer
from lightautoml.transformers.seq import GetSeqTransformer
from lightautoml.transformers.gpu.seq_gpu import SeqNumCountsTransformerGPU
from lightautoml.transformers.gpu.seq_gpu import SeqStatisticsTransformerGPU
from lightautoml.transformers.gpu.seq_gpu import GetSeqTransformerGPU

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))

@jit(nopython=True)
def gen_cols(n_cols):
    cols = [""]*n_cols
    for i in range(n_cols):
        cols[i] = "col_" + str(i)
    return cols

def gen_string_data(n, n_str):
    string_db = ["algorithm", "analog", "app", "application", "array",
                 "backup", "bandwidth", "binary", "bit", "byte"]#,
                 #"bitmap", "blog", "bookmark", "boot", "broadband",
                 #"browser" , "buffer", "bug"]
    inds = np.random.randint(0, len(string_db), (n, n_str))
    output = np.empty(inds.shape, dtype=object)
    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            output[i][j] = string_db[inds[i][j]]

    return output

def gen_data_single_target(n: int, n_num: int, n_cat: int, n_date: int,
         n_str: str, max_n_cat: int, n_ids: int, max_ids: list = None,
         cols: list = None):
    n_cols = n_num+n_cat+n_str+n_date+n_ids
    cols = gen_cols(n_cols) if cols is None else cols
    data = np.random.random((n, n_num))*100-50

    category_data = np.random.randint(0, np.random.randint(1,max_n_cat),
                                      (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000, (n, n_date))\
                               .astype(np.dtype("timedelta64[D]")) \
                              + np.datetime64("2018-01-01")

    if max_ids is None:
        id_data = np.arange(n, dtype=int)[:, np.newaxis]\
                  *np.ones(n_ids, dtype=int)[:, np.newaxis].T
        for elem in id_data.T:
            np.random.shuffle(elem)
    else:
        id_data = np.array(np.random.random((n, n_ids))*max_ids//1,
                           dtype=int)

    data = pd.DataFrame(data, columns = cols[:n_num]).astype('f')
    
    ix = [(row, col) for row in range(data.shape[0]) \
                     for col in range(data.shape[1])]
    #for row, col in random.sample(ix, int(round(.1*len(ix)))):
    #    data.iat[row, col] = np.nan
    
    nn = len(data.columns)
    for i in range(n_cat):
        data[cols[nn+i]] = pd.Series(category_data[:,i]).astype('f')
    nn = len(data.columns)
    for i in range(n_str):
        data[cols[nn+i]] = pd.Series(string_data[:,i]).astype(object)
    nn = len(data.columns)
    for i in range(n_date):
        data[cols[nn+i]] = pd.Series(date_data[:,i])
    nn = len(data.columns)
    for i in range(n_ids):
        data[cols[nn+i]] = pd.Series(id_data[:, i])

    data['TARGET'] = pd.Series(np.random.randint(0, 5, n)).astype('i')

    return 'TARGET', cols, data

if __name__ == "__main__":

    ## Data preparation
    n, n_num, n_cat, n_date, n_str = 50, 3, 2, 2, 1
    max_n_cat, n_ids = 10, 1
    cols_data1 = ["a","b","c","d","e","str1",
                  "date1", "date2", "data1_id"]
    _, _, data1 = gen_data_single_target(n, n_num, n_cat, 
                  n_date, n_str, max_n_cat, n_ids, cols=cols_data1)

    n, n_num, n_cat, n_date, n_str = 350, 2, 2, 0, 0
    max_n_cat, n_ids = 5, 1
    cols_data2 = ["h","i","j","k", "data2_id"]
    _, _, data2 = gen_data_single_target(n, n_num, n_cat, 
                  n_date, n_str, max_n_cat, n_ids, cols=cols_data2)

    max_ids = [50, 100]
    n, n_num, n_cat, n_date = 1000, 4, 6, 2
    n_str, max_n_cat, n_ids = 2, 15, 2
    target, cols, train = gen_data_single_target(n, n_num, n_cat, 
                         n_date, n_str, max_n_cat, n_ids, max_ids)

    n = 200
    _, _, test = gen_data_single_target(n, n_num, n_cat, 
                         n_date, n_str, max_n_cat, n_ids, max_ids)
    seq_params = {
             'data1':{'case': 'ids',
                      'params': {},
                      'scheme': {'to': 'plain', 
                                 'from_id': 'data1_id',
                                 'to_id': 'col_14'},
                     },
             'data2':{'case': 'ids',
                      'params': {},
                      'scheme': {'to': 'plain',
                                 'from_id': 'data2_id',
                                 'to_id': 'col_15'},
                          },
              }
    seq_data = {'data1': data1[cols_data1],
                'data2': data2[cols_data2]              
               }
    X_train = {'plain':train , 
               'seq': seq_data
              }
    X_test = {'plain':test , 
               'seq': seq_data
              }
    name = 'data2'
    #
    #
    #
    #
    #DATA1 THROWS ERROR ONLY ON GPU
    #
    #
    #
    #
    ## Data preparation finished

    task = Task('reg', metric='mae')
    task_gpu = Task('reg', metric='mae', device='gpu')
    task_mgpu = Task('reg', metric='mae', device='mgpu')
    roles={'target': target}

    reader = DictToPandasSeqReader(task=task, seq_params=seq_params)    
    res = reader.fit_read(X_train, roles=roles)
    reader_gpu = DictToCudfSeqReader(task=task_gpu,
                                    seq_params=seq_params, n_jobs=1)
    res_gpu = reader_gpu.fit_read(X_train, roles=roles)
    reader_mgpu = DictToDaskCudfSeqReader(task=task_mgpu, cv=3,
                   n_jobs = 1, npartitions=2, seq_params=seq_params)
    res_mgpu = reader_mgpu.fit_read(X_train, roles=roles)

    counts = SeqNumCountsTransformer()
    counts.fit(res.seq_data[name])
    out_counts = counts.transform(res.seq_data[name])

    counts_gpu = SeqNumCountsTransformerGPU()
    counts_gpu.fit(res_gpu.seq_data[name])
    out_counts_gpu = counts_gpu.transform(res_gpu.seq_data[name])

    counts_gpu.fit(res_mgpu.seq_data[name])
    out_counts_mgpu = counts_gpu.transform(res_mgpu.seq_data[name])

    stats = SeqStatisticsTransformer()
    stats.fit(res.seq_data[name])
    out_stats = stats.transform(res.seq_data[name])

    stats_gpu = SeqStatisticsTransformerGPU()
    stats_gpu.fit(res_gpu.seq_data[name])
    out_stats_gpu = stats_gpu.transform(res_gpu.seq_data[name])

    stats_gpu.fit(res_mgpu.seq_data[name])
    out_stats_mgpu = stats_gpu.transform(res_mgpu.seq_data[name])

    seq = GetSeqTransformer(name=name)
    seq.fit(res)
    out_seq = seq.transform(res)

    seq_gpu = GetSeqTransformerGPU(name=name)
    seq_gpu.fit(res_gpu)
    out_seq_gpu = seq_gpu.transform(res_gpu)

    seq_gpu.fit(res_mgpu)
    out_seq_mgpu = seq_gpu.transform(res_mgpu)

    assert np.allclose(out_counts.data, 
                       out_counts_gpu.data.values_host)
    assert np.allclose(out_counts.data,
                       out_counts_mgpu.data.compute().values_host)

    assert np.allclose(out_stats.data,
                       out_stats_gpu.data.values_host)

    assert np.allclose(out_stats_gpu.data.sort_index().values_host,
                       out_stats_mgpu.data.compute().sort_index().values_host)

    assert np.allclose(out_seq.data, out_seq_gpu.data.values_host)
    assert np.allclose(out_seq.data,
                       out_seq_mgpu.data.compute().values_host)










 
