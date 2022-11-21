if __name__ == "__main__":
    # import usual libraries
    import time
    import os
    import gc
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = 'mgpu'
    n_gpu = torch.cuda.device_count()
    visible_devices = ",".join([str(i) for i in range(n_gpu)])

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    # Defining a cluster with all avilable GPUs. It should be defined before importing libraries that need GPU (e.g., cupy, cudf)
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=visible_devices)
    client = Client(cluster)

    import cudf

    # import lightautoml
    from lightautoml.automl.presets.text_presets import TabularNLPAutoML
    from lightautoml.automl.presets.gpu.text_gpu_presets import TabularNLPAutoMLGPU
    from lightautoml.tasks import Task
    from lightautoml.dataset.utils import roles_parser

    # define nlp constants
    N_THREADS = 4
    N_FOLDS = 5
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TIMEOUT = 300
    TARGET_NAME = 'is_good'

    torch.set_num_threads(N_THREADS)
    torch.cuda.empty_cache()

    # load bankiru dataset
    DATASET_FULLNAME = '../data/nlp/bankiru_isgood.csv'

    # here only 1000 samples are used for time reasons (for a detailed check, one needs to use larger number:
    # 100k-500k)
    data = pd.read_csv(DATASET_FULLNAME)[["message", "title", "is_good"]].fillna("")[:1000]

    # split data
    tr_data, te_data = train_test_split(data,
            test_size=TEST_SIZE,
            stratify=data[TARGET_NAME],
            random_state=RANDOM_STATE
        )
    print(data.head())
    tr_data = pd.DataFrame(data, index=[i for i in range(tr_data.shape[0])])
    te_data = pd.DataFrame(data, index=[i for i in range(te_data.shape[0])])

    print(f'Data splitted. Parts sizes: tr_data = {tr_data.shape}, te_data = {te_data.shape}')

    # define task and roles
    task = Task('binary', device=device)

    roles = {
        'text': ['message', 'title'],
        'target': TARGET_NAME,
    }
    print(roles_parser(roles))

    def run_automl(automl, tr_data, te_data):
        t0 = time.time()
        oof_pred = automl.fit_predict(tr_data, roles=roles, verbose=1)
        t1 = time.time()
        print('Elapsed time (train): {}'.format(t1 - t0))

        t0 = time.time()
        te_pred = automl.predict(te_data)
        t1 = time.time()
        print('Elapsed time (test): {}'.format(t1 - t0))

        not_nan = np.any(~np.isnan(oof_pred.data), axis=1)
        print(f'OOF score: {roc_auc_score(tr_data[TARGET_NAME].values[not_nan], oof_pred.data[not_nan][:, 0])}')
        print(f'TEST score: {roc_auc_score(te_data[TARGET_NAME].values, te_pred.data[:, 0])}')

    # Let's manually define all parameters for nlp preset here. For different parameter configuration, one could restart
    # the script (launching different dask computations in the single process might lead to some problems)

    # tfidf parameters
    n_components = 100
    n_oversample = 0
    ngram = (1, 1)
    # required algo
    algos_gpu = ['linear_l2'] # or cb or xgb
    # text features
    text_features_gpu = 'tfidf' # or tfidf_subword or embed
    model_name = 'random_lstm'


    automl = TabularNLPAutoMLGPU(task=task,
                                  timeout=600,
                                  cpu_limit=1,
                                  #gpu_ids="0,1",
                                  gpu_ids="0",
                                  client=client, # note that client is passed
                                  general_params={
                                      'nested_cv': False,
                                      'use_algos': [algos_gpu]
                                  },
                                  reader_params={
                                      'npartitions': 2
                                  },
                                  text_params={
                                      'lang': 'ru',
                                      'verbose': False,
                                      'use_stem': False,
                                      'vocab_path': '../data/nlp/vocab_hash/bankiru_isgood_vocab_hash.txt',
                                      'is_hash': True,
                                      # 'data_path': file_name,
                                      # 'max_length': 320,
                                      # 'tokenizer': "bpe",
                                      # 'vocab_size': 31000
                                  },
                                  autonlp_params={
                                      'model_name': model_name,
                                      # 'sent_scaler': 'l1',
                                      'embedding_model': 'fasttext',
                                      'cache_dir': None
                                  },
                                  tfidf_params={
                                      'n_components': n_components,
                                      'n_oversample': n_oversample,
                                      'tfidf_params': {'ngram_range': ngram}
                                  },
                                  gbm_pipeline_params={
                                      'text_features': text_features_gpu
                                  },
                                  linear_pipeline_params={
                                      'text_features': text_features_gpu
                                  },
                                  )

    run_automl(automl, tr_data, te_data)

    client.close()
    cluster.close()


