"""
This script provides tests for NLP preprocessing transformers on dummy data:
1) test different GPU/MGPU transformer settings on artificial data
2) where applicable, compare the output of fit_predict and predict methods
3) where applicable, compare the output of transformers from CPU and GPU/MGPU versions
"""

def prepare_dummy_data(lang="en"):
    if lang == "en":
        data = pd.DataFrame({"column_1": ["Here is dummy sentence 1", "What to say?", "... No way"]*10,
                             "column_2": ["Nothing happening here.", "Say whatever you want", "Is this for real?!"]*10})
    elif lang == "ru":
        data = pd.DataFrame({"column_1": ["Всееем привет!", "Дамми предложение на русском", "Ну что сказать, ну что сказать"]*10,
                             "column_2": ["Устроены так люди", "Желают знать, желают знать", "Желают знать, что будет"]*10})
    return data


def convert_outputs(outputs):
    outputs_conv = []
    for output in outputs:
        if type(output) == scipy.sparse.csr.csr_matrix:
            outputs_conv.append(output.data)
        else:
            outputs_conv.append(output)
    return outputs_conv


def check_outputs(output_1, output_2, mode='equal', message="CPU and GPU output is similar"):
    output_1, output_2 = convert_outputs([output_1, output_2])
    if mode == 'equal':
        fl = np.array_equal(output_1, output_2)
    else:
        fl = np.allclose(output_1, output_2, atol=1e-6)
    print(f'{message}:  {fl}')


def check_transformer(transformer, dataset, transformer_name="[empty]", modes=['fit_transform', 'transform'],
                      check_fit_transform_and_transform=True):
    # fl_dask = True if isinstance(dataset, DaskCudfDataset) else False
    fl_gpu = True if isinstance(dataset, CudfDataset) else False
    for mode in modes:
        if mode == "fit":
            transformer.fit(dataset)
        elif mode == "fit_transform":
            dataset_ft = transformer.fit_transform(dataset)
            data_ft = dataset_ft.to_pandas().data if fl_gpu else dataset_ft.data
        elif mode == "transform":
            dataset_t = transformer.transform(dataset)
            data_t = dataset_t.to_pandas().data if fl_gpu else dataset_t.data

    data_ft = np.array(data_ft)
    data_t = np.array(data_t)
    assert data_ft.shape == data_t.shape, f"fit_transform shape is {data_ft.shape}, transform shape is {data_t.shape} " \
                                          f"but it should be the same"
    assert type(data_ft) == type(data_t), f"fit_transform data type is {type(data_ft)}," \
                                          f"transform data type is {type(data_t)}" \
                                          f"but it should be the same"

    if check_fit_transform_and_transform:
        assert len(modes) > 1, "One of fit_transform or transform is missing"
        check_outputs(data_ft, data_t, mode="equal", message=f"fit_transform and transform methods "
                                                             f"for {transformer_name} return the same")


def check_transformer_devices(transformers, datasets, transformer_name="[empty]", mode='equal'):
    outputs = []
    for i in range(len(transformers)):
        fl_gpu = True if isinstance(datasets[i], DaskCudfDataset) or isinstance(datasets[i], CudfDataset) else False
        output = transformers[i].fit_transform(datasets[i])
        output = output.to_pandas().data if fl_gpu else output.data
        outputs.append(np.array(output))

    check_outputs(outputs[0], outputs[1], mode=mode, message=f"{transformer_name}: CPU and GPU output is similar")
    check_outputs(outputs[0], outputs[2], mode=mode, message=f"{transformer_name}: CPU and MGPU output is similar")



if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import scipy
    import torch
    import time
    import os

    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordPiece
    from tokenizers import normalizers
    from tokenizers.normalizers import Lowercase, NFD, StripAccents
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    n_gpu = torch.cuda.device_count()
    visible_devices = ",".join([str(i) for i in range(n_gpu)])

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=visible_devices)
    client = Client(cluster)

    import cudf
    from cudf.utils.hash_vocab_utils import hash_vocab
    import dask_cudf

    from lightautoml.dataset.roles import TextRole
    from lightautoml.dataset.utils import roles_parser
    from lightautoml.dataset.np_pd_dataset import PandasDataset
    from lightautoml.dataset.gpu.gpu_dataset import CudfDataset
    from lightautoml.dataset.gpu.gpu_dataset import DaskCudfDataset

    from lightautoml.text.tokenizer import SimpleRuTokenizer
    from lightautoml.text.tokenizer import SimpleEnTokenizer
    from lightautoml.text.gpu.tokenizer_gpu import SimpleRuTokenizerGPU
    from lightautoml.text.gpu.tokenizer_gpu import SimpleEnTokenizerGPU

    from lightautoml.transformers.text import TokenizerTransformer
    from lightautoml.transformers.text import ConcatTextTransformer
    from lightautoml.transformers.text import TfidfTextTransformer
    from lightautoml.transformers.text import AutoNLPWrap
    from lightautoml.transformers.decomposition import SVDTransformer
    from lightautoml.transformers.gpu.text_gpu import TokenizerTransformerGPU
    from lightautoml.transformers.gpu.text_gpu import SubwordTokenizerTransformerGPU
    from lightautoml.transformers.gpu.text_gpu import ConcatTextTransformerGPU
    from lightautoml.transformers.gpu.text_gpu import TfidfTextTransformerGPU
    from lightautoml.transformers.gpu.text_gpu import AutoNLPWrapGPU

    df = prepare_dummy_data()
    check_roles = {
        TextRole(): df.columns
    }
    df_cudf = cudf.DataFrame(data=df)
    n_gpu = torch.cuda.device_count()
    df_daskcudf = dask_cudf.from_cudf(df_cudf, npartitions=n_gpu)#.persist()

    dataset = PandasDataset(df, roles_parser(check_roles))
    dataset_gpu = CudfDataset(df_cudf, roles_parser(check_roles))
    dataset_mgpu = DaskCudfDataset(df_daskcudf, roles_parser(check_roles))

    # Check concat transformer
    print("======================================")
    print("Checking Concat transformer")
    print("======================================")
    concat = ConcatTextTransformer()
    concat_gpu = ConcatTextTransformerGPU()
    concat_mgpu = ConcatTextTransformerGPU()

    check_transformer(concat, dataset, transformer_name="concat_cpu", modes=["fit_transform", "transform"])
    check_transformer(concat_gpu, dataset_gpu, transformer_name="concat_gpu", modes=["fit_transform", "transform"])
    check_transformer(concat_mgpu, dataset_mgpu, transformer_name="concat_mgpu", modes=["fit_transform", "transform"])

    check_transformer_devices([concat, concat_gpu, concat_mgpu],
                              [dataset, dataset_gpu, dataset_mgpu],
                              transformer_name="concat")

    # Check tokenizer transformer
    print("======================================")
    print("Checking Tokenizer transformer")
    print("======================================")
    token_ru = SimpleRuTokenizer(is_stemmer=False)
    token_ru_stem = SimpleRuTokenizer(is_stemmer=True)
    token_en = SimpleEnTokenizer(is_stemmer=False)
    token_en_stem = SimpleEnTokenizer(is_stemmer=True)

    token_ru_gpu = SimpleRuTokenizerGPU(is_stemmer=False)
    token_ru_stem_gpu = SimpleRuTokenizerGPU(is_stemmer=True)
    token_en_gpu = SimpleEnTokenizerGPU(is_stemmer=False)
    token_en_stem_gpu = SimpleEnTokenizerGPU(is_stemmer=True)

    tokens = [token_ru, token_ru_stem, token_en, token_en_stem]
    tokens_gpu = [token_ru_gpu, token_ru_stem_gpu, token_en_gpu, token_en_stem_gpu]
    for i in range(len(tokens)):
        print(f"tokenizer name: {tokens[i].__class__}, stemmer: {True if tokens[i].stemmer is not None else False}")
        tokenizer = TokenizerTransformer(tokenizer=tokens[i])
        tokenizer_gpu = TokenizerTransformerGPU(tokenizer=tokens_gpu[i])

        check_transformer(tokenizer, dataset, transformer_name="tokenizer_cpu", modes=["fit_transform", "transform"])
        check_transformer(tokenizer_gpu, dataset_gpu, transformer_name="tokenizer_gpu", modes=["fit_transform", "transform"])
        check_transformer(tokenizer_gpu, dataset_mgpu, transformer_name="tokenizer_mgpu", modes=["fit_transform", "transform"])

        check_transformer_devices([tokenizer, tokenizer_gpu, tokenizer_gpu],
                                  [dataset, dataset_gpu, dataset_mgpu],
                                  transformer_name="tokenizer")

    # Check SubwordTokenizer transformer
    print("======================================")
    print("Checking SubwordTokenizer transformer")
    print("======================================")

    # Make necessary preprocessing on a dummy data

    # Save dummy dataframe to a .txt file
    file_name = "dummy_df.txt"
    with open(file_name, "w+") as f:
        for i in range(len(df)):
            f.write(df["column_1"][i] + '\n')

    # Train tokenizer vocabulary on it
    tokenizer = 'bpe'  # or 'wordpiece'
    vocab_size = 30000
    vocab_save_path = f"{tokenizer}_{vocab_size // 1000}k_test.txt"

    if tokenizer == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]", "[SEP]", "[CLS]"]
        )
    else:
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]", "[SEP]", "[CLS]"]
        )
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train([file_name], trainer)  # train tokenizer on out .txt text data
    trained_vocab = tokenizer.get_vocab()

    # save trained vocabulary to a .txt file
    with open(vocab_save_path, 'w+') as f:
        for key in trained_vocab.keys():
            f.write(key + '\n')

    # Make hash-version of vocabulary
    vocab_save_path_hash = vocab_save_path.split('.')[0] + '_hash.txt'
    hash_vocab(vocab_save_path, vocab_save_path_hash)

    print("################ Subword tests starting ############################")
    print(f"Setting 1: vocab_path = {vocab_save_path_hash}, data_path = {None}, is_hash = {True},"
          f"max_length = {3}")
    subword_tokenizer_gpu = SubwordTokenizerTransformerGPU(vocab_path=vocab_save_path_hash, data_path=None,
                                                            is_hash=True, max_length=3)
    check_transformer(subword_tokenizer_gpu, dataset_gpu, transformer_name="subword_tokenizer_gpu", modes=["fit_transform", "transform"])
    check_transformer(subword_tokenizer_gpu, dataset_mgpu, transformer_name="subword_tokenizer_mgpu",
                      modes=["fit_transform", "transform"])
    print("####################################################################")

    print(f"Setting 2: vocab_path = {vocab_save_path}, data_path = {None}, is_hash = {False},"
          f"max_length = {30}")
    subword_tokenizer_gpu = SubwordTokenizerTransformerGPU(vocab_path=vocab_save_path, data_path=None,
                                                            is_hash=False, max_length=30)
    check_transformer(subword_tokenizer_gpu, dataset_gpu, transformer_name="subword_tokenizer_gpu",
                      modes=["fit_transform", "transform"])
    check_transformer(subword_tokenizer_gpu, dataset_mgpu, transformer_name="subword_tokenizer_mgpu",
                      modes=["fit_transform", "transform"])
    print("####################################################################")

    print(f"Setting 3: vocab_path = {None}, data_path = {file_name}, is_hash = {False},"
          f"tokenizer = {'bpe'}, vocab_size = {10}")
    subword_tokenizer_gpu = SubwordTokenizerTransformerGPU(vocab_path=None, data_path=file_name,
                                                            is_hash=False, tokenizer="bpe", vocab_size=10)
    check_transformer(subword_tokenizer_gpu, dataset_gpu, transformer_name="subword_tokenizer_gpu",
                      modes=["fit_transform", "transform"])
    check_transformer(subword_tokenizer_gpu, dataset_mgpu, transformer_name="subword_tokenizer_mgpu",
                      modes=["fit_transform", "transform"])
    print("####################################################################")

    print(f"Setting 4: vocab_path = {None}, data_path = {file_name}, is_hash = {False},"
          f"tokenizer = {'wordpiece'}, vocab_size = {20}")
    subword_tokenizer_gpu = SubwordTokenizerTransformerGPU(vocab_path=None, data_path=file_name,
                                                            is_hash=False, tokenizer="wordpiece", vocab_size=20)
    check_transformer(subword_tokenizer_gpu, dataset_gpu, transformer_name="subword_tokenizer_gpu",
                      modes=["fit_transform", "transform"])
    check_transformer(subword_tokenizer_gpu, dataset_mgpu, transformer_name="subword_tokenizer_mgpu",
                      modes=["fit_transform", "transform"])
    print("####################################################################")


    # Check tfidf + svd transformer
    print("======================================")
    print("Checking Tfidf+svd transformer")
    print("======================================")

    default_params = {"min_df": 4,
                      "max_df": 1.0,
                      "max_features": 1000,
                      "ngram_range": (1, 2)}
    n_components = 10
    tfidf = TfidfTextTransformer(default_params=default_params)
    svd = SVDTransformer(n_components=n_components)
    tfidf_gpu = TfidfTextTransformerGPU(default_params=default_params, n_components=n_components)
    tfidf_mgpu = TfidfTextTransformerGPU(default_params=default_params, n_components=n_components)

    check_transformer(tfidf, dataset, transformer_name="tfidf_cpu", modes=["fit_transform", "transform"])
    check_transformer(tfidf_gpu, dataset_gpu, transformer_name="tfidf_gpu", modes=["fit_transform", "transform"])
    check_transformer(tfidf_mgpu, dataset_mgpu, transformer_name="tfidf_mgpu", modes=["fit_transform", "transform"])


    # Check AutoNLP transformer
    print("======================================")
    print("Checking AutoNLP transformer")
    print("======================================")

    # Check models
    models = ["random_lstm", "random_lstm_bert", "borep", "pooled_bert", "wat"]
    for model in models:
        print("####################################################################")
        print(f"Model = {model}")
        autonlp = AutoNLPWrap(model_name=model)
        autonlp_gpu = AutoNLPWrapGPU(model_name=model, embedding_model="fasttext")
        autonlp_mgpu = AutoNLPWrapGPU(model_name=model, embedding_model="fasttext")

        check_transformer(autonlp, dataset, transformer_name="autonlp_cpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_gpu, dataset_gpu, transformer_name="autonlp_gpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_mgpu, dataset_mgpu, transformer_name="autonlp_mgpu", modes=["fit_transform", "transform"])

    # Check available embedding models from torch-nlp library
    embedding_models = ["fasttext", "bpe"]
    for emb_model in embedding_models:
        print("####################################################################")
        print(f"Embedding model = {emb_model}")
        autonlp = AutoNLPWrap(model_name="random_lstm")
        autonlp_gpu = AutoNLPWrapGPU(model_name="random_lstm", embedding_model=emb_model)
        autonlp_mgpu = AutoNLPWrapGPU(model_name="random_lstm", embedding_model=emb_model)

        check_transformer(autonlp, dataset, transformer_name="autonlp_cpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_gpu, dataset_gpu, transformer_name="autonlp_gpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_mgpu, dataset_mgpu, transformer_name="autonlp_mgpu", modes=["fit_transform", "transform"])

    sent_scalers = [None, "l2", "l1"]
    for sent_scaler in sent_scalers:
        print("####################################################################")
        print(f"Sent_scaler = {sent_scaler}")
        autonlp = AutoNLPWrap(model_name="random_lstm", sent_scaler=sent_scaler)
        autonlp_gpu = AutoNLPWrapGPU(model_name="random_lstm", embedding_model=emb_model, sent_scaler=sent_scaler)
        autonlp_mgpu = AutoNLPWrapGPU(model_name="random_lstm", embedding_model=emb_model, sent_scaler=sent_scaler)

        check_transformer(autonlp, dataset, transformer_name="autonlp_cpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_gpu, dataset_gpu, transformer_name="autonlp_gpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_mgpu, dataset_mgpu, transformer_name="autonlp_mgpu", modes=["fit_transform", "transform"])

    langs = ["en", "ru"]
    for lang in langs:
        print("####################################################################")
        print(f"Lang = {lang}")
        autonlp = AutoNLPWrap(model_name="random_lstm", lang=lang)
        autonlp_gpu = AutoNLPWrapGPU(model_name="random_lstm", embedding_model=emb_model, lang=lang)
        autonlp_mgpu = AutoNLPWrapGPU(model_name="random_lstm", embedding_model=emb_model, lang=lang)

        check_transformer(autonlp, dataset, transformer_name="autonlp_cpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_gpu, dataset_gpu, transformer_name="autonlp_gpu", modes=["fit_transform", "transform"])
        check_transformer(autonlp_mgpu, dataset_mgpu, transformer_name="autonlp_mgpu", modes=["fit_transform", "transform"])

    client.close()
    cluster.close()

