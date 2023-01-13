import argparse
from sklearn.model_selection import train_test_split

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
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    str_nthr = str(args.njobs)

    os.environ["OMP_NUM_THREADS"] = str_nthr
    os.environ["OPENBLAS_NUM_THREADS"] = str_nthr
    os.environ["MKL_NUM_THREADS"] = str_nthr
    os.environ["VECLIB_MAXIMUM_THREADS"] = str_nthr
    os.environ["NUMEXPR_NUM_THREADS"] = str_nthr

    from lightautoml_gpu.automl.presets.gpu.tabular_gpu_presets import TabularAutoMLGPU
    from lightautoml_gpu.tasks import Task
    from lightautoml_gpu.dataset.roles import TargetRole

    import joblib
    import numpy as np
    import torch
    import pandas as pd

    from time import time
    from sklearn.metrics import log_loss, mean_squared_error

    def cent(y_true, y_pred):

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.log(np.take_along_axis(y_pred, y_true[:, np.newaxis].astype(np.int32), axis=1)).mean()

    torch.set_num_threads(args.njobs)
    np.random.seed(args.seed)

    # paths ..
    data_info = joblib.load(os.path.join(args.bench, 'data_info.pkl'))[args.key]

    print('Train dataset {0}'.format(args.key))

    results = {}

    X_tot = joblib.load(os.path.join(args.path, data_info['data']))
    y_tot = joblib.load(os.path.join(args.path, data_info['target']))
    x_cols = ["input_" + str(i) for i in range(X_tot.shape[1])]
    y_cols = ["output_" + str(i) for i in range(y_tot.shape[1])]

    data = pd.DataFrame(np.concatenate([X_tot, y_tot], axis=1),
                        columns=x_cols + y_cols)
    X_tot = None
    y_tot = None
    print(data.head())
    target_columns = y_cols

    train, test = train_test_split(data, test_size=0.2, random_state=args.seed)
    data = None
    task_type = 'multi:reg' if data_info['task_type'] == 'multitask' else data_info['task_type']
    loss = 'mse' if task_type == 'multi:reg' else 'logloss'
    automl = TabularAutoMLGPU(task=Task(task_type, loss=loss,
                                        device="gpu"),
                              timeout=args.timeout,
                              config_path=args.config,)

    roles = {TargetRole(): target_columns}

    # TRAIN
    t = time()
    oof_predictions = automl.fit_predict(train.reset_index().drop(['index'], axis=1), roles=roles, verbose=4)
    results['train_time'] = time() - t

    # VALID
    t = time()
    test_pred = automl.predict(test.reset_index().drop(['index'], axis=1)).data
    results['prediction_time'] = time() - t

    # EVALUATE
    if type(test_pred) is not np.ndarray:
        test_pred = test_pred.get()
    if data_info['task_type'] == 'multilabel':
        results['score'] = log_loss(test[target_columns].values, test_pred, eps=1e-7)
    if data_info['task_type'] == 'multitask':
        results['score'] = mean_squared_error(test[target_columns].values, test_pred)

    print(results)

    automl.to_cpu()
    cpu_inf = automl.predict(test.reset_index().drop(['index'], axis=1)).data
    print("cpu_inf vs test_pred")
    print(cpu_inf)
    print(test_pred)

    from joblib import dump
    import time
    pickle_file = './gpu.joblib'
    start = time.time()
    with open(pickle_file, 'wb') as f:
        dump(automl, f)
    raw_dump_duration = time.time() - start
    print("Raw dump duration: %0.3fs" % raw_dump_duration)
