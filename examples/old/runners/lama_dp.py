import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--bench', type=str)
parser.add_argument('-p', '--path', type=str)

parser.add_argument('-k', '--key', type=str)
parser.add_argument('-f', '--fold', type=int)

parser.add_argument('-n', '--njobs', type=int)
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-d', '--device', type=str)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-t', '--timeout', type=int)

if __name__ == '__main__':
    
    import os
    from time import sleep
    args = parser.parse_args()
    
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    str_nthr = str(args.njobs)
    
    os.environ["OMP_NUM_THREADS"] = str_nthr # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = str_nthr # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = str_nthr # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = str_nthr # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = str_nthr # export NUMEXPR_NUM_THREADS=6
    
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster


    cluster = LocalCUDACluster(rmm_managed_memory=True,
                          protocol="ucx", enable_nvlink=True,
                          threads_per_worker=1, memory_limit="30GB",
                          rmm_pool_size="5GB")

    client = Client(cluster)

    from lightautoml.automl.presets.gpu.tabular_gpu_presets import TabularAutoML_gpu
    from lightautoml.tasks import Task
    from lightautoml.dataset.roles import TargetRole
    
    import joblib
    import numpy as np
    import torch
    import pandas as pd
    import cudf
    
    
    from time import time, sleep
    from sklearn.metrics import roc_auc_score, mean_squared_error
    
    def cent(y_true, y_pred):

        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)

        return -np.log(np.take_along_axis(y_pred, y_true[:, np.newaxis].astype(np.int32), axis=1)).mean()
    
    torch.set_num_threads(args.njobs)
    np.random.seed(args.seed)
    
    # paths .. 
    data_info = joblib.load(os.path.join(args.bench, 'data_info.pkl'))[args.key]
    folds = joblib.load(os.path.join(args.bench, 'folds', '{0}.pkl'.format(args.key)))
    
    print('Train dataset {0}, fold {1}'.format(args.key, args.fold))
    
    # GET DATA AND PREPROCESS
    
    read_csv_params = {}
    if 'read_csv_params' in data_info:
        read_csv_params = {**read_csv_params, **data_info['read_csv_params']}
        print(read_csv_params)
        
    data = pd.read_csv(os.path.join(args.path, data_info['path']), **read_csv_params)
    
    if 'drop' in data_info:
        data.drop(data_info['drop'], axis=1, inplace=True)
    
    if 'class_map' in data_info:
        data[data_info['target']] = data[data_info['target']].map(data_info['class_map']).values
        assert data[data_info['target']].notnull().all(), 'Class mapping is set unproperly'
    
    print(data.head())
    
    results = {}
    
    # CREATE AUTOML

    client.run(cudf.set_allocator, "managed")
    cudf.set_allocator("managed")

    automl = TabularAutoML_gpu(task=Task(data_info['task_type'], device="mgpu"), 
                               timeout=args.timeout,
                               config_path=args.config,
                               client=client)

    roles = {TargetRole(): data_info['target']}

    # TRAIN
    t = time()
    oof_predictions = automl.fit_predict(data[folds!=args.fold].reset_index(drop=True),
                                         roles=roles, verbose=4)
    results['train_time'] = time() - t

    # VALID

    t = time()
    test_pred = automl.predict(data[folds==args.fold].reset_index(drop=True)).data
    results['prediction_time'] = time() - t

    print(data[folds==args.fold].shape)
    print(test_pred.shape)

    # EVALUATE

    if type(test_pred) is not np.ndarray:
        test_pred = test_pred.get()

    if data_info['task_type'] == 'binary':
        results['score'] = roc_auc_score(data[folds==args.fold][data_info['target']].values, test_pred[:, 0])

    if data_info['task_type'] == 'reg':
        results['score'] = mean_squared_error(data[folds==args.fold][data_info['target']].values, test_pred[:, 0])

    if data_info['task_type'] == 'multiclass':
        results['score'] = cent(data[folds==args.fold][data_info['target']].values, test_pred)

    print(results)

    exit(0)
    
