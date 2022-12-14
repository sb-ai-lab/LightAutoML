import numpy as np
import pandas as pd
import dask_cudf
import cudf
import cupy as cp
import random
import string
from numba import jit

from time import perf_counter

from lightautoml_gpu.tasks import Task

from lightautoml_gpu.reader.gpu.cudf_reader import CudfReader
from lightautoml_gpu.reader.gpu.daskcudf_reader import DaskCudfReader
from lightautoml_gpu.reader.base import PandasToPandasReader

import os
import time

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))

N_THREADS = 8 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 600 # Time in seconds for automl run
TARGET_NAME = 'TARGET' # Target column name

@jit(nopython=True)
def gen_cols(n_cols):
    cols = [""]*n_cols
    for i in range(n_cols):
        cols[i] = "col_" + str(i)
    return cols

def gen_string_data(n, n_str):
    string_db = ["algorithm", "analog", "app", "application", "array",
                 "backup", "bandwidth", "binary", "bit", "byte",
                 "bitmap", "blog", "bookmark", "boot", "broadband",
                 "browser" , "buffer", "bug"]
    inds = np.random.randint(0, len(string_db), (n, n_str))
    output = np.empty(inds.shape, dtype=object)
    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            output[i][j] = string_db[inds[i][j]]

    return output

def generate_data(n, n_num, n_cat, n_date, n_str, max_n_cat):
    n_cols = n_num+n_cat+n_str+n_date
    cols = gen_cols(n_cols)
    data = np.random.random((n, n_num))*100-50

    category_data = np.random.randint(0, 
                           np.random.randint(1,max_n_cat), (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000,
                               (n, n_date)).astype(np.dtype("timedelta64[D]")) \
                               + np.datetime64("2018-01-01")

    data = pd.DataFrame(data, columns = cols[:n_num]).astype('f')

    ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
    for row, col in random.sample(ix, int(round(.1*len(ix)))):
        data.iat[row, col] = np.nan

    nn = len(data.columns)
    for i in range(n_cat):
        data[cols[nn+i]] = pd.Series(category_data[:,i]).astype('f')
    nn = len(data.columns)
    for i in range(n_str):
        data[cols[nn+i]] = pd.Series(string_data[:,i]).astype(object)
    nn = len(data.columns)
    for i in range(n_date):
        data[cols[nn+i]] = pd.Series(date_data[:,i])

    data['TARGET'] = pd.Series(np.random.randint(0, 4, n)).astype('i')

    return 'TARGET', cols, data


adv_roles = True
roles = {'target': 'TARGET'}

task = Task("reg")
task_gpu = Task("reg", device="gpu")
task_mgpu = Task("reg", device="mgpu")

reader = PandasToPandasReader(task, advanced_roles=adv_roles)

gpu_reader = CudfReader(task_gpu, advanced_roles=adv_roles, n_jobs=2)

dd_reader = DaskCudfReader(task_mgpu, advanced_roles=adv_roles,
                           n_jobs=2, index_ok=True, npartitions=2)

target, _, data = generate_data(n=20000, n_num=15, n_cat=10, n_date=5,
                                       n_str=15, max_n_cat=50)

data = data.copy()

st = perf_counter()
ds = reader.fit_read(data, roles = roles)
print(perf_counter() - st)
st = perf_counter()
gpu_ds = gpu_reader.fit_read(data, roles = roles)
print(perf_counter() - st)
st = perf_counter()
dd_ds = dd_reader.fit_read(data, roles = roles)
print(perf_counter() - st)

print(ds.data.shape)
print(gpu_ds.data.shape)
print(dd_ds.data.compute().shape)
