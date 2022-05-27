import logging
import os

from html import escape
from numbers import Number
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd


try:
    import gensim
except:
    import warnings

    warnings.warn("'gensim' - package isn't installed")

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ...pipelines.features.text_pipeline import _tokenizer_by_lang
from ...text.utils import seed_everything
from .data_process import get_embedding_matrix
from .data_process import get_len_dataloader
from .data_process import get_len_dataset
from .data_process import get_tokenized
from .data_process import get_vocab
from .data_process import map_tokenized_to_id
from .l2x_model import L2XModel
from .utils import WrappedTokenizer
from .utils import WrappedVocabulary
from .utils import cross_entropy_multiple_class
from .utils import draw_html


logger = logging.getLogger(__name__)


class L2XTextExplainer:
    """
    Learning-to-Explain method for explaining NLP models for classification and regression,
    selecting the most informative (important) tokens in document.

    This method optimizes the variational lower bound on mutual information
    between target and selected number of tokens. Number of selected
    tokens is constant (n_important).

    The main differences from original model are:

    1. Dynamic batching, using the length of tokenized document.
    Without it the model can highlight unknown tokens (<UNK>).

    2. There are original and softsub methods for sampling important tokens.

    3. Additional regularizer-like term to encourage highliting neighborhood tokens.

    More precisely explanations of how the methods work in:
    1) Learning to Explain: An Information-Theoretic
    Perspective on Model Interpretation, J. Chen et al.
    https://arxiv.org/abs/1802.07814

    2) Reparameterizable Subset Sampling via Continuous Relaxations, S. Xie, S. Ermon.
    https://arxiv.org/abs/1901.10517

    Additional info:

    1. After traning all models will be on cpu and in evaluation mode.


    How should it works:

        >>> task = Task('reg')
        >>> automl = TabularNLPAutoML(task=task,
        >>>     timeout=600, gpu_ids = '0',
        >>>     general_params = {'nested_cv': False, 'use_algos': [['nn']]},
        >>>     text_params = {'lang': 'ru'})
        >>> automl.fit_predict(train, roles=roles)
        >>> from lightautoml.addons.interpretation import L2XTextExplainer
        >>> l2x = L2XTextExlpainer(automl, tokenizer='ru', device='cuda:0', n_important=10)
        >>> l2x.fit(train2, cols_to_explain='col1') # there can be also list of columns
        >>> explanations = l2x['col1'].explain_instances(data, batch_size=1)
        >>> explanations.get_all() # return list of (tokens_i, mask_i). The mask and tokens arrays have same sizes.
        >>> explanations[i].visualize_in_notebook() # produces the visualization of highlited tokens in document i.

    """

    def __init__(
        self,
        automl,
        tokenizer: Optional[Union[str, Callable]] = None,
        train_device: str = "cpu",
        inference_device: str = "cpu",
        verbose: bool = True,
        binning_mode: str = "linear",
        bins_number: int = 3,
        n_important: int = 10,
        learning_rate: float = 1e-4,
        n_epochs: int = 200,
        optimizer: Type[torch.optim.Optimizer] = Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        patience: int = 0,
        extreme_patience: int = 0,
        train_batch_size: int = 64,
        valid_batch_size: int = 16,
        temperature: float = 2,
        temp_anneal_factor: float = 0.985,
        conv_filters: int = 100,
        conv_ksize: int = 3,
        hidden_dim: int = 100,
        drop_rate: float = 0.2,
        importance_sampler: str = "softsub",
        embedder: Optional[Union[str, Dict[str, np.ndarray]]] = None,
        embedding_dim: Optional[int] = None,
        trainable_embeds: bool = False,
        max_vocab_length: int = -1,
        gamma: float = 0.01,
        gamma_anneal_factor: float = 1.0,
        random_seed: int = 42,
        deterministic: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """

        Args:
            automl: Automl object.
            tokenizer: String with language ['ru', 'en'],
                or callable function that take something,
                that deal with sentence in string to list of tokens.
                of list of strings. If None the lang
                from automl's text_params will be used.
            train_device: Device that will be used for traning L2X.
                Name of device should be valid for torch.device.
            inference_device: Device that will be used for inference L2X.
                Name of device should be valid for torch.device.
            verbose: Verbose training.
            binning_mode: For dynamic batching we use binning sampler by number
                of tokens in the tokenized document. This parameter specifies
                the method for constructing the boundaries for binning.
                Valid parameter values: 'linear' (min-max binning,
                like linspace), 'hist' (histogram binning).
            bins_number: Number of bins.
            n_important: Number of important tokens.
            learning_rate: Learning rate of optimizer for traning L2X.
            n_epochs: Number of epochs for training L2X.
            optimizer: Should be optimizer in pytorch format.
            optimizer_params: Additional params of optimizer,
                exclude learning_rate.
            patience: Number of epochs before reducing learning rate.
            extreme_patience: Early stopping epochs.
            train_batch_size: Size of batch for training process.
            valid_batch_size: Size of batch for validation process.
            temperature: Temperature of concrete distribution sampling.
            temp_anneal_factor: Annealing temperature. The temperature will be
                multiplied by this coefficient after every epoch.
            conv_filters: Number of convolution kernels in model,
                that produces important tokens.
            conv_ksize: Size of convolution kernel.
            hidden_dim: Size of fully connected layer in L2X.
            drop_rate: Dropout rates in L2X.
            importance_sampler: Specifices method of sampling importance.
            embedder: Embedding dictionary or path to fasttext/dict of embeddings.
            trainable_embeds: To train embeddings of L2X.
            max_vocab_length: Maximum vocabulary length. If -1 then include all in train set.
            gamma: Special coefficient, that encourage neighborhood of important tokens.
            gamma_anneal_factor: Annealing gamma. The gamma will be
                multiplied by this coefficient after every epoch.
            random_seed: Random seed.
            deterministic: Use cuda deterministic.
            cache_dir: Directory used for checkpointing model for early stopping.
                By default, it will infer from automl cache directory,
                or './l2x_cache' in case there is no opportunity to infer.
        """
        self.automl = automl
        self.reader = automl.reader
        self.task_name = automl.reader.task.name
        self.roles = automl.reader.roles

        if self.task_name == "binary":
            self._loss = nn.BCELoss()
            self.n_outs = 1
        elif self.task_name == "reg":
            self._loss = nn.MSELoss()
            self.n_outs = 1
        elif self.task_name == "multiclass":
            self._loss = cross_entropy_multiple_class
            self.n_outs = self.reader._n_classes

        if isinstance(tokenizer, str):
            if tokenizer not in ["ru", "en"]:
                raise ValueError("Tokenizer must be one 'ru' or 'en', but {} given".format(tokenizer))
            self._tokenizer = _tokenizer_by_lang[tokenizer](is_stemmer=False)
            self.tokenizer = WrappedTokenizer(self._tokenizer)
        elif tokenizer is None:
            # check is_stemmer=True or False,
            # who's the best and what easy to implement
            lang = automl.text_params["lang"]
            self._tokenizer = _tokenizer_by_lang[lang](is_stemmer=False)
            self.tokenizer = WrappedTokenizer(self._tokenizer)
        elif callable(tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError("Unkown type of tokenizer: {}".format(type(tokenizer)))

        self.train_device = torch.device(train_device)
        self.inference_device = torch.device(inference_device)
        self.verbose = verbose

        if binning_mode not in ["linear", "hist"]:
            raise ValueError("Only avaliable 'linear', 'hist' binning mods, but {} given".format(binning_mode))
        self.binning_mode = binning_mode
        self.bins_number = bins_number
        self.k = n_important
        self.learning_rate = learning_rate
        if n_epochs <= 0:
            raise ValueError("Epochs number should be positive, but {} given".format(n_epochs))
        self.n_epochs = n_epochs

        if not issubclass(optimizer, torch.optim.Optimizer):
            raise TypeError("Not torch.optim.Optimizer like optimizer format, {} given".format(type(optimizer)))
        self.optimizer = optimizer
        optimizer_params = optimizer_params or {}
        self.optim_params = {**optimizer_params, "lr": self.learning_rate}
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        if temperature <= 0:
            raise ValueError("Temperature should be positive, but {} given".format(temperature))
        self.T = temperature

        if temp_anneal_factor <= 0:
            raise ValueError("Temperature annealing factor should be positive, but {} given".format(temp_anneal_factor))
        self.temp_anneal_factor = temp_anneal_factor

        if conv_filters <= 0:
            raise ValueError(
                "Number of filters in convolution layers should be positive, but {} given".format(conv_filters)
            )
        self.conv_filters = conv_filters
        if conv_ksize <= 0:
            raise ValueError(
                "Kernel size of filters in convolution layers should be positive, but {} given".format(conv_ksize)
            )
        self.conv_ksize = conv_ksize
        if hidden_dim <= 0:
            raise ValueError("Dimention of hidden layer should be positive, but {} given".format(hidden_dim))
        self.hidden_dim = hidden_dim
        if drop_rate >= 1 or drop_rate < 0:
            raise ValueError("Dropout rate should be in [0, 1), but {} given".format(drop_rate))
        self.drop_rate = drop_rate

        if importance_sampler not in ["gumbeltopk", "softsub"]:
            raise ValueError(
                "Only possible values for importance samper are 'gumbeltopk',"
                "'softsub', but {} given".format(importance_sampler)
            )
        self.importance_sampler = importance_sampler

        if embedder is None:
            self.embedder = None
            if embedding_dim is None:
                raise ValueError("At least embedding_dim or embedder should be not none")
            self.embedding_dim = embedding_dim
        elif isinstance(embedder, str):
            try:
                self.embedder = gensim.models.FastText.load(embedder).wv
            except:
                try:
                    self.embedder = gensim.models.FastText.load_fasttext_format(embedder).wv
                except:
                    self.embedder = gensim.models.KeyedVectors.load(embedder).wv
            self.embedding_dim = self.embedder.vector_size
        elif isinstance(embedder, dict):
            self.embedder = embedder
            self.embedding_dim = next(iter(embedder.values())).shape[0]
        elif isinstance(embedder, gensim.models.KeyedVectors):
            self.embedder = embedder
            self.embedding_dim = self.embedder.vector_size
        else:
            raise TypeError("Unknown embedder type: {}".format(embedder))
        self.trainable_embeds = trainable_embeds

        if not isinstance(max_vocab_length, int):
            raise TypeError("max_vocab_length should be int, but {} given".format(type(max_vocab_length)))
        elif max_vocab_length < -1 or max_vocab_length == 0:
            raise ValueError(
                "Only avaliable values for max_vocab_length: -1 or grater 0, but {} given".format(max_vocab_length)
            )
        self.max_vocab_length = max_vocab_length

        if gamma < 0:
            logger.info2("For now sparse token highlighting will be encouraged, since gamma < 0")
        self.gamma = gamma
        if gamma_anneal_factor <= 0:
            raise ValueError("Gamma annealing factor should be positive, but {} given".format(temp_anneal_factor))
        self.gamma_anneal_factor = gamma_anneal_factor
        self.explainers = {}
        self.random_seed = random_seed
        self.deterministic = deterministic
        seed_everything(random_seed, deterministic)

        if extreme_patience == 0:
            extreme_patience = n_epochs
        if patience == 0:
            patience = extreme_patience
        if patience > extreme_patience:
            logger.info2(
                "extreme_patience (={}) must be greater or equal patience (={}), now extreme_patience also ={}".format(
                    extreme_patience, patience, patience
                )
            )
            self.extreme_patience = patience
        self.patience = patience
        self.extreme_patience = extreme_patience

        if cache_dir is None:
            cache_dir = automl.autonlp_params["cache_dir"] or "./l2x_cache"
        if not isinstance(cache_dir, str):
            raise TypeError("Unknown type for cache_dir: {}".format(type(cache_dir)))
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self._checkpoint_path = cache_dir + "/l2x_checkpoint.pt"

    @property
    def n_important(self):
        return self.k

    @property
    def temperature(self):
        return self.T

    def fit(
        self,
        train_data: pd.DataFrame,
        valid_data: Optional[pd.DataFrame] = None,
        cols_to_explain: Optional[Union[str, List[str]]] = None,
    ):
        """
        Fit model for all columns in cols_to_explain.

        Target for L2X is the predictions of AutoML model. So, that you
        can pass any dataset there, but it recommend using training
        or validation datasets for the initial AutoML model.

        Args:
            train_data: Dataset, used for training the L2X model.
            valid_data: Validation dataset.
            cols_to_explain: Columns, for which the models will be built.
                If none the L2X models will be built corresponding to each
                textual column.

        """
        cols = self._get_cols(cols_to_explain)
        train_preds = self.automl.predict(train_data).data
        valid_preds = None
        if valid_data is not None:
            valid_preds = self.automl.predict(valid_data).data
        for col in cols:
            self.explainers[col] = self._fit_one(col, train_data, train_preds, valid_data, valid_preds)

    def _get_cols(self, cols_to_explain: Union[None, str, List[str]]) -> List[str]:
        """
        Handler for column names.

        Args:
            cols_to_explain: Explaining columns.

        Returns:
            Handled column names.

        """
        if isinstance(cols_to_explain, str):
            cols_to_explain = [cols_to_explain]
        if cols_to_explain is None:
            cols_to_explain = self.reader.cols_by_type("Text")
        else:
            for col in cols_to_explain:
                if self.roles[col].name != "Text":
                    raise ValueError("Column {} is not text column".format(col))

        return cols_to_explain

    def _fit_one(
        self,
        col_to_explain: str,
        train_data: pd.DataFrame,
        train_preds: np.ndarray,
        valid_data: Union[None, pd.DataFrame],
        valid_preds: Union[None, np.ndarray],
    ) -> "_L2XExplainer":
        """

        Args:
            col_to_explain: Explaining column.
            train_data: Dataset for training.
            train_preds: AutoML's (or any different model) predictions
                for train dataset.
            valid_data: Dataset for validation.
            valid_preds: AutoML's (or any different model) predictions
                for validation dataset.

        Returns:
            Explanation object.

        """
        train_tokenized = get_tokenized(train_data, col_to_explain, self.tokenizer)
        word_to_id, id_to_word = get_vocab(train_tokenized, self.max_vocab_length)
        word_to_id = WrappedVocabulary(word_to_id)
        weights_matrix = get_embedding_matrix(id_to_word, self.embedder, self.embedding_dim)
        train_data = map_tokenized_to_id(train_tokenized, word_to_id, self.k)
        train_dataset = get_len_dataset(train_data, train_preds)
        data_lens = self._get_lens(train_dataset)
        boundaries = self.calc_bins(data_lens, train_preds)
        train_dataloader = get_len_dataloader(train_dataset, self.train_batch_size, boundaries, mode="train")
        valid_dataloader = None
        if valid_data is not None:
            valid_tokenized = get_tokenized(valid_data, col_to_explain, self.tokenizer)
            valid_data = map_tokenized_to_id(valid_tokenized, word_to_id, self.k)
            valid_dataset = get_len_dataset(valid_data, valid_preds)
            valid_dataloader = get_len_dataloader(valid_dataset, self.valid_batch_size, boundaries, mode="valid")

        model = L2XModel(
            task_name=self.task_name,
            n_outs=self.n_outs,
            voc_size=len(word_to_id),
            embed_dim=self.embedding_dim,
            conv_filters=self.conv_filters,
            conv_ksize=self.conv_ksize,
            drop_rate=self.drop_rate,
            hidden_dim=self.hidden_dim,
            T=self.temperature,
            k=self.n_important,
            weights_matrix=weights_matrix,
            trainable_embeds=self.trainable_embeds,
            sampler=self.importance_sampler,
            anneal_factor=self.temp_anneal_factor,
        )
        train_loss_logs = []
        if valid_data is not None:
            valid_loss_logs = []
        self.train(model, train_dataloader, train_loss_logs, valid_dataloader, valid_loss_logs)

        return _L2XExplainer(
            model,
            col_to_explain,
            boundaries,
            self.tokenizer,
            word_to_id,
            id_to_word,
            self.inference_device,
            self.k,
            self.task_name,
            train_loss_logs,
            valid_loss_logs,
        )

    def train(
        self,
        model: L2XModel,
        train_dataloader: torch.utils.data.DataLoader,
        train_loss_logs: List[float],
        valid_dataloader: Optional[torch.utils.data.DataLoader] = None,
        valid_loss_logs: List[Union[float, None]] = None,
    ):
        """
        Trainer for L2X.

        Args:
            model: Model to train.
            train_dataloader: Dataloader, used for training.
            valid_dataloader: Dataloader, used for validation.

        """
        loss = self._loss
        optimizer = self.optimizer(model.parameters(), **self.optim_params)
        model.to(self.train_device)
        prev_best = -1
        best_loss = torch.finfo(float).max
        gamma = self.gamma

        scheduler = ReduceLROnPlateau(optimizer, patience=self.patience)
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(model, train_dataloader, loss, optimizer, self.train_device, gamma)
            valid_loss = self._validate(model, valid_dataloader, loss, self.train_device)
            train_loss_logs.append(train_loss)
            if valid_loss_logs is not None:
                valid_loss_logs.append(valid_loss)
            if self.verbose:
                if valid_loss is None:
                    logger.info3("Epoch: {}/{}, train loss: {}".format(epoch + 1, self.n_epochs, train_loss))
                else:
                    logger.info3(
                        "Epoch: {}/{}, train loss: {}, valid loss: {}".format(
                            epoch + 1, self.n_epochs, train_loss, valid_loss
                        )
                    )
            if valid_loss is not None:
                scheduler.step(valid_loss)
                if self.extreme_patience > 0 and valid_loss < best_loss:
                    best_loss = valid_loss
                    prev_best = epoch
                    torch.save(model.state_dict(), self._checkpoint_path)

                elif self.extreme_patience > 0 and epoch - prev_best > self.extreme_patience:
                    model.load_state_dict(torch.load(self._checkpoint_path))
                    break
            if epoch != self.n_epochs - 1:
                model.anneal()
                gamma *= self.gamma_anneal_factor

        model.cpu()

    def _train_epoch(
        self,
        model: L2XModel,
        train_dataloader: torch.utils.data.DataLoader,
        criterion: Union[torch.nn.modules.loss._Loss, Callable],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float,
    ) -> float:
        """
        Train only one epoch.

        Args:
            model: Trained L2X model.
            train_dataloader: Dataloader, used for training.
            loss: Torch loss object. Unnormalized (biased) negative log-likelihood.
            optimizer: Optimizer that should be one or nothing.

        """
        model.train()
        if self.verbose:
            train_dataloader = tqdm(train_dataloader, desc="train", disable=False)

        accum_loss = 0.0
        iters = 0
        for data in train_dataloader:
            x = data["text"]
            x = x.to(device)
            y = data["target"]
            y = y.to(device)
            optimizer.zero_grad()
            pred, corr_pred = model(x)
            # Negative loglikelihood up to a constant
            nll_loss = criterion(pred, y)
            # Encouragement of neighbour tokens
            corr_loss = torch.mean(((corr_pred[:, 1:]) ** 2 * (corr_pred[:, :-1]) ** 2).sum(-1))
            # Not sure that optima of this pair of losses is the same
            # but dunno how get best validation score
            loss = nll_loss - gamma * corr_loss
            loss.backward()
            optimizer.step(),
            nll_loss = nll_loss.data.cpu().detach().numpy()
            accum_loss += nll_loss
            iters += 1

            if self.verbose:
                train_dataloader.set_description("train nll (loss={:.4f})".format(accum_loss / iters))

        return accum_loss / iters

    def _validate(
        self,
        model: L2XModel,
        valid_dataloader: Union[None, torch.utils.data.DataLoader],
        criterion: Union[torch.nn.modules.loss._Loss, Callable],
        device: torch.device,
    ) -> float:
        if valid_dataloader is None:
            return
        model.eval()
        model.to(device)
        loss = 0.0
        iters = 0
        with torch.no_grad():
            for data in valid_dataloader:
                x = data["text"]
                x = x.to(device)
                y = data["target"]
                y = y.to(device)
                pred, _ = model(x)
                nll_loss = criterion(pred, y).data.cpu().detach().numpy()
                loss += nll_loss
                iters += 1

        return loss / iters

    @staticmethod
    def _get_lens(dataset) -> np.ndarray:
        lens = []
        for inst in dataset:
            lens.append(inst["len"])
        return np.array(lens)

    def calc_bins(self, lens, target) -> np.ndarray:
        if self.binning_mode == "linear":
            return self._linear_bins(lens)
        elif self.binning_mode == "hist":
            return self._hist_bins(lens)
        elif self.bin_mode == "mi":
            return self._mi_bins(lens, target)

    def _linear_bins(self, lens) -> np.ndarray:
        return np.r_[0, np.linspace(lens.min(), lens.max() + 1, self.bins_number + 1)[1:]]

    def _hist_bins(self, lens) -> np.ndarray:
        return np.r_[0, np.hist(lens, self.bins_number)[1]]

    def _mi_bins(self, data, target) -> np.ndarray:
        raise NotImplementedError("Mutual information binning is not avaliable")

    def __getitem__(self, col) -> np.ndarray:
        if col not in self.explainers:
            raise ValueError("There is no explanation for {}".format(col))
        return self.explainers[col]


class _L2XExplainer:
    def __init__(
        self,
        model: L2XModel,
        col_to_explain: str,
        bins: List[int],
        tokenizer: Callable,
        word_to_id: Dict[str, int],
        id_to_word: Dict[int, str],
        inference_device: torch.device,
        n_important: int,
        task_name: str,
        train_loss_logs: List[float],
        valid_loss_logs: List[Union[float, None]],
    ):
        """
        Args:
            model: L2X explanation model.
            col_to_explain: Explaining column.
            bins: Binning used for training.
            tokenizer: Tokenizer function.
            embedder: Embedding dictionary (token->embedding).
            inference_device: Inference device.
            n_important: Number of important tokens.
            task_name: Task name.
            train_loss_logs: Train process logs.
                Contains train loss list.
            valid_loss_logs: Valid process logs.
                Contains valid loss list.

        """
        self.model = model
        self.col_to_explain = col_to_explain
        self.bins = bins
        self.tokenizer = tokenizer
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.inference_device = inference_device
        self.k = n_important
        self.task_name = task_name
        self.train_loss_logs = train_loss_logs
        self.valid_loss_logs = valid_loss_logs

    @property
    def train_loss(self):
        return self.train_loss_logs

    @property
    def valid_loss(self):
        return self.valid_loss_logs

    @property
    def n_train_epochs(self):
        return len(self.train_loss_logs)

    @property
    def n_important(self):
        return self.k

    def explain_instances(self, data: pd.DataFrame, batch_size: int = 1) -> "L2XExplanationsContainer":
        """
        Get explanations for data.

        Args:
            data: Data to explain.
            batch_size: Size of batch. Carefully, if batch_size > 0
                every document will be padded to max size of documents in abstract.

        """
        data_tokenized = get_tokenized(data, self.col_to_explain, self.tokenizer)
        data = map_tokenized_to_id(data_tokenized, self.word_to_id, self.k)
        dataset = get_len_dataset(data, np.ones((len(data), 1)))

        dataloader = get_len_dataloader(dataset, batch_size, mode="test")
        masks = self._get_masks(dataloader)  # care the order.
        tokens = [self._tokens_to_text(dataset[i]["text"]) for i in range(len(dataset))]

        return L2XExplanationsContainer(tokens, masks, self.task_name)

    def _tokens_to_text(self, tokens):
        return list(map(self.id_to_word.get, tokens.numpy()))

    def _get_masks(self, dataloader: torch.utils.data.DataLoader) -> List[List[Number]]:
        important_tokens = []
        self.model.eval()
        self.model.to(self.inference_device)
        with torch.no_grad():
            for data in dataloader:
                x = data["text"]
                x = x.to(self.inference_device)
                _, imp_mask = self.model(x)
                important_tokens.extend([x.tolist() for x in imp_mask.cpu().numpy()])
            return important_tokens


class L2XExplanation:
    """
    Class for visualizing important tokens for single document.
    """

    _hightliting_color: str = "#6DB5F5"
    _background_color: str = "#FFFFFF"

    def __init__(self, tokens: List[str], mask: List[Number], task_name: str):
        """
        Args:
            tokens: Tokens of tokenized document.
            mask: Mask for important tokens. Non zero elements
                correspond to importatnt tokens.
            task_name: Task name.

        """
        if len(tokens) != len(mask):
            raise ValueError("Dimention mismatch for tokens and mask")
        self.tokens = tokens
        self.mask = mask
        self.task_name = task_name

    def visualize_in_notebook(self):
        """
        Visualization of important tokens in notebook.
        """
        from IPython.display import HTML
        from IPython.display import display_html

        token_weights = [(escape(x), y) for x, y in zip(self.tokens, self.mask)]
        html_code = draw_html(token_weights, self.task_name, self._hightliting_color, grad_line=False)
        display_html(HTML(html_code))


class L2XExplanationsContainer:
    """
    Container of explanations.
    """

    def __init__(self, docs: List[List[str]], masks: List[List[float]], task_name: str):
        """

        Args:
            docs: Tokenized documents.
            masks: Mask for importances.

        """
        self.docs = docs
        self.masks = masks
        self.task_name = task_name

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return L2XExplanation(self.docs[i], self.masks[i], self.task_name)

    def get_all(self):
        return [(x, y) for x, y in zip(self.docs, self.masks)]
