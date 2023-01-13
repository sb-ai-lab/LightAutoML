import numpy as np
import pandas as pd
import random
import string
from numba import jit
import cupy as cp
from time import perf_counter

from lightautoml_gpu.tasks import Task

from lightautoml_gpu.reader.gpu.daskcudf_reader import DaskCudfReader
from lightautoml_gpu.transformers.base import SequentialTransformer
from lightautoml_gpu.transformers.gpu import numeric_gpu, categorical_gpu, datetime_gpu
from lightautoml_gpu.transformers import numeric, categorical, datetime

from lightautoml_gpu.pipelines.utils import get_columns_by_role

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))

N_THREADS = 8  # threads cnt for lgbm and linear models
N_FOLDS = 5  # folds cnt for AutoML
RANDOM_STATE = 42  # fixed random state for various reasons
TEST_SIZE = 0.2  # Test size for metric check
TIMEOUT = 600  # Time in seconds for automl run
TARGET_NAME = 'TARGET'  # Target column name


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


def generate_data(n, n_num, n_cat, n_date, n_str, max_n_cat):
    n_cols = n_num + n_cat + n_str + n_date
    cols = gen_cols(n_cols)
    data = np.random.random((n, n_num)) * 100 - 50

    category_data = np.random.randint(0, np.random.randint(1, max_n_cat), (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000, (n, n_date)).astype(np.dtype("timedelta64[D]")) \
        + np.datetime64("2018-01-01")

    data = pd.DataFrame(data, columns=cols[:n_num]).astype('f')

    ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
    for row, col in random.sample(ix, int(round(.1 * len(ix)))):
        data.iat[row, col] = np.nan

    nn = len(data.columns)
    for i in range(n_cat):
        data[cols[nn + i]] = pd.Series(category_data[:, i]).astype('f')
    nn = len(data.columns)
    for i in range(n_str):
        data[cols[nn + i]] = pd.Series(string_data[:, i]).astype(object)
    nn = len(data.columns)
    for i in range(n_date):
        data[cols[nn + i]] = pd.Series(date_data[:, i])

    # data['TARGET'] = pd.Series(np.random.randint(0, 4, n)).astype('i')
    data['TARGET'] = pd.Series(np.random.random(n) * 3)

    return 'TARGET', cols, data


n = 100000
roles = {'target': 'TARGET'}
target, _, data = generate_data(n=n, n_num=20, n_cat=20, n_date=5,
                                n_str=10, max_n_cat=20)

data = data.copy()


data['TARGET'] = pd.Series(np.random.randint(0, 4, n)).astype('i')
task = Task('multiclass', device="mgpu")
reader = DaskCudfReader(task, advanced_roles=False,
                        n_jobs=1, index_ok=True,
                        npartitions=2)

dd_ds = reader.fit_read(data, roles=roles)
gpu_ds = dd_ds.to_cudf()
ds = dd_ds.to_pandas()

cats = ds[:, get_columns_by_role(ds, 'Category')]
gpu_cats = gpu_ds[:, get_columns_by_role(gpu_ds, 'Category')]
dd_cats = dd_ds[:, get_columns_by_role(dd_ds, 'Category')]

trf = SequentialTransformer(
    [categorical.LabelEncoder(), categorical.MultiClassTargetEncoder()]
)
gpu_trf = SequentialTransformer(
    [categorical_gpu.LabelEncoderGPU(),
     categorical_gpu.MultiClassTargetEncoderGPU()]
)

print("multiclasstarget encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

print("multiclasstarget encoder cpu, gpu and mgpu outputs:")
print(enc.data)
print()
print(enc_gpu.data)
print()
print(enc_mgpu.data.compute().values_host)

data['TARGET'] = pd.Series(np.random.random(n) * 3)
task_mgpu = Task("reg", device="mgpu")
reader = DaskCudfReader(task_mgpu, advanced_roles=False,
                        n_jobs=1, compute=True, index_ok=True,
                        npartitions=2)

dd_ds = reader.fit_read(data, roles=roles)
gpu_ds = dd_ds.to_cudf()
ds = dd_ds.to_pandas()

trf = categorical.LabelEncoder()
gpu_trf = categorical_gpu.LabelEncoderGPU()

cats = ds[:, get_columns_by_role(ds, 'Category')]
gpu_cats = gpu_ds[:, get_columns_by_role(gpu_ds, 'Category')]
dd_cats = dd_ds[:, get_columns_by_role(dd_ds, 'Category')]

print("label encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = SequentialTransformer(
    [categorical.LabelEncoder(), categorical.TargetEncoder()]
)
gpu_trf = SequentialTransformer(
    [categorical_gpu.LabelEncoderGPU(), categorical_gpu.TargetEncoderGPU()]
)

print("target encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = categorical.FreqEncoder()
gpu_trf = categorical_gpu.FreqEncoderGPU()

print("freq encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = categorical.OrdinalEncoder()
gpu_trf = categorical_gpu.OrdinalEncoderGPU()

print("ordinal encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = SequentialTransformer(
    [categorical.LabelEncoder(), categorical.OHEEncoder(make_sparse=False)]
)
gpu_trf = SequentialTransformer(
    [categorical_gpu.LabelEncoderGPU(), categorical_gpu.OHEEncoderGPU(make_sparse=False)]
)

print("ohee encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = SequentialTransformer(
    [categorical.LabelEncoder(), categorical.CatIntersectstions()]
)
gpu_trf = SequentialTransformer(
    [categorical_gpu.LabelEncoderGPU(), categorical_gpu.CatIntersectionsGPU()]
)

print("catintersections encoder:")
st = perf_counter()
enc = trf.fit_transform(cats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_cats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_cats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

print("catintersections cpu, gpu, mgpu outputs:")
print(enc.data)
print()
print(enc_gpu.data)
print()
print(enc_mgpu.data.compute().values_host)

###############################################################
###############################################################
###############################################################

dats = ds[:, get_columns_by_role(ds, 'Datetime')]
gpu_dats = gpu_ds[:, get_columns_by_role(gpu_ds, 'Datetime')]
dd_dats = dd_ds[:, get_columns_by_role(dd_ds, 'Datetime')]

trf = datetime.TimeToNum()
gpu_trf = datetime_gpu.TimeToNumGPU()

print("timetonum encoder:")
st = perf_counter()
enc = trf.fit_transform(dats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_dats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_dats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = datetime.BaseDiff(base_names=[dats.features[0]], diff_names=[dats.features[1]])
gpu_trf = datetime_gpu.BaseDiffGPU(base_names=[dats.features[0]], diff_names=[dats.features[1]])

print("basediff encoder:")
st = perf_counter()
enc = trf.fit_transform(dats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_dats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_dats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = datetime.DateSeasons()
gpu_trf = datetime_gpu.DateSeasonsGPU()

print("dateseasons encoder:")
st = perf_counter()
enc = trf.fit_transform(dats)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_dats)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_dats)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

###############################################################
###############################################################
###############################################################

nums = ds[:, get_columns_by_role(ds, 'Numeric')]
gpu_nums = gpu_ds[:, get_columns_by_role(gpu_ds, 'Numeric')]
dd_nums = dd_ds[:, get_columns_by_role(dd_ds, 'Numeric')]

trf = numeric.NaNFlags()
gpu_trf = numeric_gpu.NaNFlagsGPU()

print("nanflags encoder:")
st = perf_counter()
enc = trf.fit_transform(nums)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_nums)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_nums)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)

trf = numeric.FillnaMedian()
gpu_trf = numeric_gpu.FillnaMedianGPU()

print("fillnamedian encoder:")
st = perf_counter()
enc = trf.fit_transform(nums)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_nums)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_nums)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))
assert np.allclose(enc.data, enc_mgpu.data.compute().values_host)


trf = numeric.FillInf()
gpu_trf = numeric_gpu.FillInfGPU()

print("fillinf encoder:")
st = perf_counter()
enc = trf.fit_transform(nums)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_nums)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_nums)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(np.nan_to_num(enc.data),
                   np.nan_to_num(cp.asnumpy(enc_gpu.data)))
assert np.allclose(np.nan_to_num(enc.data),
                   np.nan_to_num(enc_mgpu.data.compute().values_host))

trf = numeric.LogOdds()
gpu_trf = numeric_gpu.LogOddsGPU()

print("logodds encoder:")
st = perf_counter()
enc = trf.fit_transform(nums)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_nums)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_nums)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(np.nan_to_num(enc.data),
                   np.nan_to_num(cp.asnumpy(enc_gpu.data)))
assert np.allclose(np.nan_to_num(enc.data),
                   np.nan_to_num(enc_mgpu.data.compute().values_host))

trf = numeric.StandardScaler()
gpu_trf = numeric_gpu.StandardScalerGPU()

print("standard scaler encoder:")
st = perf_counter()
enc = trf.fit_transform(nums)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_nums)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_nums)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

print("standard scaler cpu, gpu, mgpu outputs:")
print(enc.data)
print()
print(enc_gpu.data)
print()
print(enc_mgpu.data.compute().values_host)

trf = numeric.QuantileBinning()
gpu_trf = numeric_gpu.QuantileBinningGPU()


print("quantile binning encoder:")
st = perf_counter()
enc = trf.fit_transform(nums)
print("cpu: ", perf_counter() - st)
st = perf_counter()
enc_gpu = gpu_trf.fit_transform(gpu_nums)
print("gpu: ", perf_counter() - st)
st = perf_counter()
enc_mgpu = gpu_trf.fit_transform(dd_nums)
print("mgpu: ", perf_counter() - st)
st = perf_counter()

assert np.allclose(enc.data, cp.asnumpy(enc_gpu.data))

print("quantile binning cpu, mgpu outputs:")
print(enc.data)
print()
print(enc_mgpu.data.compute().values_host)
