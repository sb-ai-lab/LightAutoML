from copy import copy, deepcopy
from typing import List, Optional, Dict, Tuple, Any, Union

import gensim
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer as PysparkRegexTokenizer
from pyspark.ml.feature import Tokenizer as PysparkTokenizer, CountVectorizer, IDF, Normalizer
from pyspark.sql.functions import concat_ws
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from sklearn.base import TransformerMixin

from lightautoml.dataset.roles import TextRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.base import ObsoleteSparkTransformer
from lightautoml.text.dl_transformers import DLTransformer
from lightautoml.text.tokenizer import BaseTokenizer
from lightautoml.text.tokenizer import SimpleEnTokenizer
from lightautoml.transformers.text import TunableTransformer, model_by_name, logger
from lightautoml.transformers.text import text_check


class TfidfTextTransformer(ObsoleteSparkTransformer, TunableTransformer):
    """Simple Tfidf vectorizer."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tfidf"
    _default_params = {
        "min_df": 5,
        "max_df": 1.0,
        "max_features": 30_000,
        "dtype": np.float32,
        "normalization": 2.0
    }

    # These properties are not supported
    # cause there is no analogues in Spark ML
    # "ngram_range": (1, 1),
    # "analyzer": "word",

    @property
    def features(self) -> List[str]:
        """Features list."""

        return self._features

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        subs: Optional[float] = None,
        random_state: int = 42,
    ):
        """

        Args:
            default_params: algo hyperparams.
            freeze_defaults: Flag.
            subs: Subsample to calculate freqs. If ``None`` - full data.
            random_state: Random state to take subsample.

        Note:
            The behaviour of `freeze_defaults`:

            - ``True`` :  params may be rewritten depending on dataset.
            - ``False``:  params may be changed only
              manually or with tuning.

        """
        super().__init__(default_params, freeze_defaults)
        self.subs = subs
        self.random_state = random_state
        self.idf_columns_pipelines: Optional[Dict[str, Tuple[PipelineModel, str, int]]] = None
        self.vocab_size: Optional[int] = None

    def init_params_on_input(self, dataset: SparkDataset) -> dict:
        """Get transformer parameters depending on dataset parameters.

        Args:
            dataset: Dataset used for model parmaeters initialization.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        suggested_params = copy(self.default_params)
        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        # TODO: decide later what to do with this part
        # rows_num = len(dataset.data)
        # if rows_num > 50_000:
        #     suggested_params["min_df"] = 25

        return suggested_params

    def _fit(self, dataset: SparkDataset):
        """Fit tfidf vectorizer.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            self.

        """

        # convert to accepted dtype and get attributes
        if self._params is None:
            self.params = self.init_params_on_input(dataset)

        sdf = dataset.data
        if self.subs:
            sdf = sdf.sample(self.subs, seed=self.random_state)
        sdf = sdf.fillna("")

        self.idf_columns_pipelines = dict()
        feats = []
        for c in dataset.features:
            # TODO: set params here from self.params
            tokenizer = PysparkTokenizer(inputCol=c, outputCol=f"{c}_words")
            count_tf = CountVectorizer(
                minDF=self.params["min_df"],
                maxDF=self.params["max_df"],
                vocabSize=self.params["max_features"],
                inputCol=tokenizer.getOutputCol(),
                outputCol=f"{c}_word_features"
            )
            out_col = f"{self._fname_prefix}__{c}"
            idf = IDF(inputCol=count_tf.getOutputCol(), outputCol=f"{c}_idf_features")

            stages = [tokenizer, count_tf, idf]
            if self.params["normalization"] and self.params["normalization"] > 0:
                norm = Normalizer(inputCol=idf.getOutputCol(), outputCol=out_col, p=self.params["normalization"])
                stages.append(norm)

            pipeline = Pipeline(stages=stages)
            tfidf_pipeline_model = pipeline.fit(sdf)

            # features = list(
            #     np.char.array([self._fname_prefix + "_"])
            #     + np.arange(count_tf.getVocabSize()).astype(str)
            #     + np.char.array(["__" + c])
            # )
            # feats.extend(features)

            self.idf_columns_pipelines[c] = (tfidf_pipeline_model, out_col, count_tf.getVocabSize())

        self._features = feats

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform text dataset to sparse tfidf representation.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Sparse dataset with encoded text.

        """

        sdf = dataset.data
        sdf = sdf.fillna("")

        # transform

        curr_sdf = sdf
        all_idf_features = []
        all_idf_roles = []
        for c in dataset.features:
            tfidf_model, idf_col, vocab_size = self.idf_columns_pipelines[c]

            curr_sdf = tfidf_model.transform(curr_sdf)

            role = NumericVectorOrArrayRole(
                size=vocab_size,
                element_col_name_template=f"{self._fname_prefix}_{{}}__{idf_col}"
            )

            all_idf_features.append(idf_col)
            all_idf_roles.append(role)

        # all_idf_features = [
        #     vector_to_array(F.col(idf_col))[i].alias(feat)
        #     for idf_col, features in idf2features.items()
        #     for i,feat in enumerate(features)
        # ]

        new_sdf = curr_sdf.select(*dataset.service_columns, *all_idf_features)

        output = dataset.empty()
        output.set_data(new_sdf, self._features, all_idf_roles)

        return output


class AutoNLPWrap(ObsoleteSparkTransformer):
    """Calculate text embeddings."""

    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "emb"
    fasttext_params = {"vector_size": 64, "window": 3, "min_count": 1}
    _names = {"random_lstm", "random_lstm_bert", "pooled_bert", "wat", "borep"}
    _trainable = {"wat", "borep", "random_lstm"}

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        model_name: str,
        embedding_model: Optional[str] = None,
        cache_dir: str = "./cache_NLP",
        bert_model: Optional[str] = None,
        transformer_params: Optional[Dict] = None,
        subs: Optional[int] = None,
        multigpu: bool = False,
        random_state: int = 42,
        train_fasttext: bool = False,
        fasttext_params: Optional[Dict] = None,
        fasttext_epochs: int = 2,
        sent_scaler: Optional[str] = None,
        verbose: bool = False,
        device: Any = "0",
        **kwargs: Any,
    ):
        """

        Args:
            model_name: Method for aggregating word embeddings
              into sentence embedding.
            transformer_params: Aggregating model parameters.
            embedding_model: Word level embedding model with dict
              interface or path to gensim fasttext model.
            cache_dir: If ``None`` - do not cache transformed datasets.
            bert_model: Name of HuggingFace transformer model.
            subs: Subsample to calculate freqs. If None - full data.
            multigpu: Use Data Parallel.
            random_state: Random state to take subsample.
            train_fasttext: Train fasttext.
            fasttext_params: Fasttext init params.
            fasttext_epochs: Number of epochs to train.
            verbose: Verbosity.
            device: Torch device or str.
            **kwargs: Unused params.

        """
        if train_fasttext:
            assert model_name in self._trainable, f"If train fasstext then model must be in {self._trainable}"

        assert model_name in self._names, f"Model name must be one of {self._names}"
        self.device = device
        self.multigpu = multigpu
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.subs = subs
        self.train_fasttext = train_fasttext
        self.fasttext_epochs = fasttext_epochs
        if fasttext_params is not None:
            self.fasttext_params.update(fasttext_params)
        self.sent_scaler = sent_scaler
        self.verbose = verbose
        self.model_name = model_name
        self.transformer_params = model_by_name[self.model_name]
        if transformer_params is not None:
            self.transformer_params.update(transformer_params)

        self._update_bert_model(bert_model)
        if embedding_model is not None:
            if isinstance(embedding_model, str):
                try:
                    embedding_model = gensim.models.FastText.load(embedding_model)
                except:
                    try:
                        embedding_model = gensim.models.FastText.load_fasttext_format(embedding_model)
                    except:
                        embedding_model = gensim.models.KeyedVectors.load(embedding_model)

            self.transformer_params = self._update_transformers_emb_model(self.transformer_params, embedding_model)

        else:

            self.train_fasttext = self.model_name in self._trainable

        if self.model_name == "wat":
            raise NotImplementedError("WeightedAverageTransformer is not yet supported, but it is planned")
            # self.transformer = WeightedAverageTransformer
        else:
            self.transformer = DLTransformer

        self.column_transformers: Optional[Dict[str, Tuple[TransformerMixin, NumericVectorOrArrayRole, str]]] = None

    def _update_bert_model(self, bert_model: str):
        if bert_model is not None:
            if "dataset_params" in self.transformer_params:
                self.transformer_params["dataset_params"]["model_name"] = bert_model
            if "embedding_model_params" in self.transformer_params:
                self.transformer_params["embedding_model_params"]["model_name"] = bert_model
            if "model_params" in self.transformer_params:
                self.transformer_params["model_params"]["model_name"] = bert_model
        return self

    def _update_transformers_emb_model(
        self, params: Dict, model: Any, emb_size: Optional[int] = None
    ) -> Dict[str, Any]:
        if emb_size is None:
            try:
                # Gensim checker [1]
                emb_size = model.vector_size
            except:
                try:
                    # Gensim checker[2]
                    emb_size = model.vw.vector_size
                except:
                    try:
                        # Natasha checker
                        emb_size = model[model.vocab.words[0]].shape[0]
                    except:
                        try:
                            # Dict of embeddings checker
                            emb_size = next(iter(model.values())).shape[0]
                        except:
                            raise ValueError("Unrecognized embedding dimention, please specify it in model_params")
        try:
            model = model.wv
        except:
            pass

        if self.model_name == "wat":
            params["embed_size"] = emb_size
            params["embedding_model"] = model
        elif self.model_name in {"random_lstm", "borep"}:
            params["dataset_params"]["embedding_model"] = model
            params["dataset_params"]["embed_size"] = emb_size
            params["model_params"]["embed_size"] = emb_size

        return params

    def _fit(self, dataset: SparkDataset):
        """Fit chosen transformer and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        """

        # TODO: cache should be reimplemented after the discussion
        # if self.cache_dir is not None:
        #     if not os.path.exists(self.cache_dir):
        #         os.makedirs(self.cache_dir)
        # set transformer features

        # convert to accepted dtype and get attributes
        # dataset = dataset.to_pandas()
        # df = dataset.data

        sdf = dataset.data

        # fit ...
        # TODO: replace subs with implementation in the base class
        # if self.subs is not None and df.shape[0] >= self.subs:
        #     subs = df.sample(n=self.subs, random_state=self.random_state)
        # else:
        #     subs = df

        names = []
        self.column_transformers = dict()
        for c in sdf.columns:
            transformer_params = deepcopy(self.transformer_params)

            # TODO: cannot implement embedding training using FastText this way
            # TODO: need to have a pytorch based version for training
            # if self.train_fasttext:
            #     embedding_model = gensim.models.FastText(**self.fasttext_params)
            #     common_texts = [i.split(" ") for i in subs[i].values]
            #     embedding_model.build_vocab(corpus_iterable=common_texts)
            #     embedding_model.train(
            #         corpus_iterable=common_texts,
            #         total_examples=len(common_texts),
            #         epochs=self.fasttext_epochs,
            #     )
            #     transformer_params = self._update_transformers_emb_model(transformer_params, embedding_model)

            transformer = self.transformer(
                verbose=self.verbose,
                device=self.device,
                multigpu=self.multigpu,
                **transformer_params,
            )
            emb_name = transformer.get_name()
            emb_size = transformer.get_out_shape()

            role = NumericVectorOrArrayRole(size=emb_size,
                                            element_col_name_template=f"{self._fname_prefix}_{emb_name}_{{}}__{c}",
                                            dtype=np.float32,
                                            is_vector=False)

            feat_name = f"{self._fname_prefix}_{emb_name}__{c}"

            # TODO: the column transformer has to support working with dataframes,
            # TODO: e.g. there should be special Spark adapted version of this column transformer
            fitted_col_transformer = transformer.fit(sdf.select(c))

            self.column_transformers[c] = (fitted_col_transformer, role, feat_name)

            logger.info3(f"Feature {c} fitted")

            names.append(feat_name)

            # TODO: probably we should do that
            # del transformer
            # gc.collect()
            # torch.cuda.empty_cache()

        self._features = names
        return self

    # TODO: this method may be unified with what we have in AutoCVWrap and ImageFeaturesTransformer
    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform tokenized dataset to text embeddings.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Numpy dataset with text embeddings.

        """

        sdf = dataset.data

        # transform
        new_cols = []
        roles = []
        for c in sdf.columns:
            # TODO: should be replaced with new implementation after a discussion
            # if self.cache_dir is not None:
            #     full_hash = get_textarr_hash(df[i]) + get_textarr_hash(self.dicts[i]["feats"])
            #     fname = os.path.join(self.cache_dir, full_hash + ".pkl")
            #     if os.path.exists(fname):
            #         logger.info3(f"Load saved dataset for {i}")
            #         with open(fname, "rb") as f:
            #             new_arr = pickle.load(f)
            #     else:
            #         new_arr = self.dicts[i]["transformer"].transform(df[i])
            #         with open(fname, "wb") as f:
            #             pickle.dump(new_arr, f)
            # else:
            trans, role, out_col_name = self.column_transformers[c]
            transformer_bcast = dataset.spark_session.sparkContext.broadcast(trans)

            @pandas_udf("array<float>", PandasUDFType.SCALAR)
            def calculate_embeddings(data: pd.Series) -> pd.Series:
                transformer = transformer_bcast.value
                embeds = pd.Series(list(transformer.transform(data)))
                return embeds

            new_cols.append(calculate_embeddings(c).alias(out_col_name))
            roles.append(role)

            logger.info3(f"Feature {c} transformed")

        new_sdf = sdf.select(new_cols)

        output = dataset.empty()
        output.set_data(new_sdf, self.features, roles)

        return output

    @staticmethod
    def _sentence_norm(x: np.ndarray, mode: Optional[str] = None) -> Union[np.ndarray, float]:
        """Get sentence embedding norm."""
        if mode == "l2":
            return ((x ** 2).sum(axis=1, keepdims=True)) ** 0.5
        elif mode == "l1":
            return np.abs(x).sum(axis=1, keepdims=True)
        if mode is not None:
            logger.info2("Unknown sentence scaler mode: sent_scaler={}, " "no normalization will be used".format(mode))
        return 1


class Tokenizer(ObsoleteSparkTransformer):
    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "tokenized"

    _can_unwind_parents = False

    def __init__(self, tokenizer: BaseTokenizer = SimpleEnTokenizer()):
        """
        Args:
            tokenizer: text tokenizer.
        """
        self.tokenizer = tokenizer

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        spark_data_frame = dataset.data

        for i, column in enumerate(dataset.features):
            # PysparkTokenizer transforms strings to lowercase, do not use it
            # tokenizer = PysparkTokenizer(inputCol=column, outputCol=self._fname_prefix + "__" + column)

            tokenizer = PysparkRegexTokenizer(inputCol=column, outputCol=self._fname_prefix + "__" + column, toLowercase=False)
            tokenized = tokenizer.transform(spark_data_frame)
            spark_data_frame = tokenized
            spark_data_frame = spark_data_frame.drop(column)

        output = dataset.empty()
        output.set_data(spark_data_frame, self.features, TextRole(np.str))

        return output


class ConcatTextTransformer(ObsoleteSparkTransformer):
    _fit_checks = (text_check,)
    _transform_checks = ()
    _fname_prefix = "concated"

    _can_unwind_parents = False

    def __init__(self, special_token: str = " [SEP] "):
        """

        Args:
            special_token: Add special token between columns.

        """
        self.special_token = special_token

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        spark_data_frame = dataset.data
        spark_column_names = dataset.features

        colum_name = self._fname_prefix + "__" + "__".join(spark_column_names)
        concatExpr = concat_ws(self.special_token, *spark_column_names).alias(colum_name)
        concated = spark_data_frame.select(*dataset.service_columns, concatExpr)

        output = dataset.empty()
        output.set_data(concated, self.features, TextRole(np.str))

        return output
