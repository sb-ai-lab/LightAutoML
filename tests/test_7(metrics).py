from lightautoml_gpu.tasks.common_metric import _valid_str_binary_metric_names as bin_cpu
from lightautoml_gpu.tasks.common_metric import _valid_str_reg_metric_names as reg_cpu
from lightautoml_gpu.tasks.common_metric import _valid_str_multiclass_metric_names as multi_cpu

from lightautoml_gpu.tasks.gpu.common_metric_gpu import _valid_str_binary_metric_names_gpu as bin_gpu
from lightautoml_gpu.tasks.gpu.common_metric_gpu import _valid_str_reg_metric_names_gpu as reg_gpu
from lightautoml_gpu.tasks.gpu.common_metric_gpu import _valid_str_multiclass_metric_names_gpu as multi_gpu


from lightautoml_gpu.tasks.gpu import common_metric_gpu as metric_gpu

import numpy as np
import pandas as pd

import cupy as cp
import cudf

import dask.array as da
import dask_cudf

size = (100, 3)

y_multi = np.random.randint(0,3, size[0])
y_multi_cp = cp.asarray(y_multi)
y_multi_cudf = cudf.DataFrame(y_multi_cp)
y_multi_da = dask_cudf.from_cudf(y_multi_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.int32))

y_mul = np.random.randint(0,3, size[0])
y_mul_cp = cp.asarray(y_mul)
y_mul_cudf = cudf.DataFrame(y_mul_cp)
y_mul_da = dask_cudf.from_cudf(y_mul_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.int32))

y_bin = np.random.randint(0,2, size[0])
y_bin_cp = cp.asarray(y_bin)
y_bin_cudf = cudf.DataFrame(y_bin_cp)
y_bin_da = dask_cudf.from_cudf(y_bin_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.int32))

y_one = np.random.random(size[0])
y_one_cp = cp.asarray(y_one)
y_one_cudf = cudf.DataFrame(y_one_cp)
y_one_da = dask_cudf.from_cudf(y_one_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.float32))

w_one = np.random.random(size[0])
w_one_cp = cp.asarray(w_one)
w_one_cudf = cudf.DataFrame(w_one_cp)
w_one_da = dask_cudf.from_cudf(w_one_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.float32))

y_true = np.random.random(size)
y_pred = np.random.random(size)
y_pred = y_pred/y_pred.sum(axis=1, keepdims=1)
weight = np.random.random(size)

y_true_cp = cp.asarray(y_true)
y_pred_cp = cp.asarray(y_pred)
weight_cp = cp.asarray(weight)

y_true_cudf = cudf.DataFrame(y_true_cp)
y_pred_cudf = cudf.DataFrame(y_pred_cp)
weight_cudf = cudf.DataFrame(weight_cp)

y_true_da = dask_cudf.from_cudf(y_true_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.float32))
y_pred_da = dask_cudf.from_cudf(y_pred_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.float32))
weight_da = dask_cudf.from_cudf(weight_cudf, npartitions=2)\
           .to_dask_array(lengths=True, meta=cp.array((), 
                          dtype=cp.float32))

##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = bin_cpu['auc'](y_bin, y_one)#, sample_weight=w_one)
res_cp = bin_gpu['auc'](y_bin_cp, y_one_cp)#, w_one_cp)
res_da = bin_gpu['auc'](y_bin_da, y_one_da)#, w_one_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = bin_cpu['logloss'](y_bin, y_one, sample_weight=w_one)
res_cp = bin_gpu['logloss'](y_bin_cp, y_one_cp, w_one_cp)
res_da = bin_gpu['logloss'](y_bin_da, y_one_da, w_one_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = bin_cpu['accuracy'](y_bin, y_one)#, w_one)
res_cp = bin_gpu['accuracy'](y_bin_cp, y_one_cp)#, w_one_cp)
res_da = bin_gpu['accuracy'](y_bin_da, y_one_da)#, w_one_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = reg_cpu['r2'](y_one, w_one)#, sample_weight=w_one)
res_cp = reg_gpu['r2'](y_one_cp, w_one_cp)#, w_one_cp)
res_da = reg_gpu['r2'](y_one_da, w_one_da)#, w_one_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-1
##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = reg_cpu['mse'](y_true, y_pred)#, sample_weight=weight)
res_cp = reg_gpu['mse'](y_true_cp, y_pred_cp)#, weight_cp)
res_da = reg_gpu['mse'](y_true_da, y_pred_da)#, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = reg_cpu['mae'](y_true, y_pred)#, sample_weight = weight)
res_cp = reg_gpu['mae'](y_true_cp, y_pred_cp)#, weight_cp)
res_da = reg_gpu['mae'](y_true_da, y_pred_da)#, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = reg_cpu['rmsle'](y_true, y_pred, weight)
res_cp = reg_gpu['rmsle'](y_true_cp, y_pred_cp, weight_cp)
res_da = reg_gpu['rmsle'](y_true_da, y_pred_da, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = reg_cpu['fair'](y_true, y_pred, weight)
res_cp = reg_gpu['fair'](y_true_cp, y_pred_cp, weight_cp)
res_da = reg_gpu['fair'](y_true_da, y_pred_da, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = reg_cpu['huber'](y_true, y_pred, weight)
res_cp = reg_gpu['huber'](y_true_cp, y_pred_cp, weight_cp)
res_da = reg_gpu['huber'](y_true_da, y_pred_da, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = reg_cpu['quantile'](y_true, y_pred, weight)
res_cp = reg_gpu['quantile'](y_true_cp, y_pred_cp, weight_cp)
res_da = reg_gpu['quantile'](y_true_da, y_pred_da, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = reg_cpu['mape'](y_true, y_pred, weight)
res_cp = reg_gpu['mape'](y_true_cp, y_pred_cp, weight_cp)
res_da = reg_gpu['mape'](y_true_da, y_pred_da, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = multi_cpu['auc_mu'](y_multi, y_pred, weight)
res_cp = multi_gpu['auc_mu'](y_multi_cp, y_pred_cp, weight_cp)
res_da = multi_gpu['auc_mu'](y_multi_da, y_pred_da, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = multi_cpu['auc'](y_multi, y_pred)#, weight)
res_cp = multi_gpu['auc'](y_multi_cp, y_pred_cp)#, weight_cp)
res_da = multi_gpu['auc'](y_multi_da, y_pred_da)#, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-1
##############################################################
#DOESN"T SUPPORT WEIGHTS FOR NOW
res = multi_cpu['crossentropy'](y_multi, y_pred)#, sample_weight=w_one)
res_cp = multi_gpu['crossentropy'](y_multi_cp, y_pred_cp)#, weight_cp)
res_da = multi_gpu['crossentropy'](y_multi_da, y_pred_da)#, weight_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2
##############################################################
res = multi_cpu['accuracy'](y_multi, y_pred, w_one)
res_cp = multi_gpu['accuracy'](y_multi_cp, y_pred_cp, w_one_cp)
res_da = multi_gpu['accuracy'](y_multi_da, y_pred_da, w_one_da)
assert abs(res - res_cp) < 1e-1
assert abs(res - res_da) < 1e-1
##############################################################
res = multi_cpu['f1_macro'](y_multi, y_pred, w_one)
res_cp = multi_gpu['f1_macro'](y_multi_cp, y_pred_cp, w_one_cp)
res_da = multi_gpu['f1_macro'](y_multi_da, y_pred_da, w_one_da)
assert abs(res - res_cp) < 1e-2 
assert abs(res - res_da) < 1e-2

