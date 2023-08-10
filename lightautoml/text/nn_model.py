"""Neural Net modules for differen data types."""

import logging

from typing import Any, List, Tuple, Type
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import operator

try:
    from transformers import AutoModel
except:
    import warnings

    warnings.warn("'transformers' - package isn't installed")

from ..tasks.base import Task
from .dl_transformers import pooling_by_name


logger = logging.getLogger(__name__)


class UniversalDataset:
    """Dataset class for mixed data.

    Args:
        data: Dict with data.
        y: Array of target variable.
        w: Optional array of observation weight.
        tokenizer: Transformers tokenizer.
        max_length: Max sentence length.
        stage: Name of current training / inference stage.

    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 256,
        stage: str = "test",
    ):
        self.data = data
        self.y = y
        self.w = w
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage = stage

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        res = {"label": self.y[index]}
        res.update({key: value[index] for key, value in self.data.items() if key != "text"})
        if (self.tokenizer is not None) and ("text" in self.data):
            sent = self.data["text"][index, 0]  # only one column
            _split = sent.split("[SEP]")
            sent = _split if len(_split) == 2 else (sent,)
            data = self.tokenizer.encode_plus(
                *sent, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True
            )

            res.update({i: np.array(data[i]) for i in data.keys()})
        if self.w is not None:
            res["weight"] = self.w[index]

        return res


class Clump(nn.Module):
    """Clipping input tensor.

    Args:
        min_v: Min value.
        max_v: Max value.

    """

    def __init__(self, min_v: int = -50, max_v: int = 50):
        super(Clump, self).__init__()

        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = torch.clamp(x, self.min_v, self.max_v)
        return x


class TextBert(nn.Module):
    """Text data model.

    Class for working with text data based on HuggingFace transformers.

    Args:
        model_name: Transformers model name.
        pooling: Pooling type.

    Note:
        There are different pooling types:

            - cls: Use CLS token for sentence embedding
                from last hidden state.
            - max: Maximum on seq_len dimension for non masked
                inputs from last hidden state.
            - mean: Mean on seq_len dimension for non masked
                inputs from last hidden state.
            - sum: Sum on seq_len dimension for non masked
                inputs from last hidden state.
            - none: Without pooling for seq2seq models.

    """

    _poolers = {"cls", "max", "mean", "sum", "none"}

    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "cls"):
        super(TextBert, self).__init__()
        if pooling not in self._poolers:
            raise ValueError("pooling - {} - not in the list of available types {}".format(pooling, self._poolers))

        self.transformer = AutoModel.from_pretrained(model_name)
        self.n_out = self.transformer.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU(inplace=True)
        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-pass."""
        # last hidden layer
        encoded_layers, _ = self.transformer(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            token_type_ids=inp.get("token_type_ids"),
            return_dict=False,
        )

        # pool the outputs into a vector
        encoded_layers = self.pooling(encoded_layers, inp["attention_mask"].unsqueeze(-1).bool())
        mean_last_hidden_state = self.activation(encoded_layers)
        mean_last_hidden_state = self.dropout(mean_last_hidden_state)
        return mean_last_hidden_state


class CatEmbedder(nn.Module):
    """Category data model.

    Args:
        cat_dims: Sequence with number of unique categories
            for category features.
        emb_dropout: Dropout probability.
        emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        max_emb_size: Max embedding size.

    """

    def __init__(
        self, cat_dims: Sequence[int], emb_dropout: bool = 0.1, emb_ratio: int = 3, max_emb_size: int = 50, **kwargs
    ):
        super(CatEmbedder, self).__init__()
        emb_dims = [(int(x), int(min(max_emb_size, max(1, (x + 1) // emb_ratio)))) for x in cat_dims]
        self.no_of_embs = sum([y for x, y in emb_dims])
        assert self.no_of_embs != 0, "The input is empty."
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dropout_layer = nn.Dropout(emb_dropout) if emb_dropout else nn.Identity()

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            Int with module output shape.

        """
        return self.no_of_embs

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-pass."""
        output = torch.cat(
            [emb_layer(inp["cat"][:, i]) for i, emb_layer in enumerate(self.emb_layers)],
            dim=1,
        )
        output = self.emb_dropout_layer(output)
        return output


class ContEmbedder(nn.Module):
    """Numeric data model.

    Class for working with numeric data.

    Args:
        num_dims: Sequence with number of numeric features.
        input_bn: Use 1d batch norm for input data.

    """

    def __init__(self, num_dims: int, input_bn: bool = True, **kwargs):
        super(ContEmbedder, self).__init__()
        self.n_out = num_dims
        self.bn = nn.Identity()
        if input_bn:
            self.bn = nn.BatchNorm1d(num_dims)
        assert num_dims != 0, "The input is empty."

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.n_out

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-pass."""
        output = inp["cont"]
        output = self.bn(output)
        return output


class BasicEmbedding(nn.Module):
    """A basic embedding that creates an embedded vector for each field value from https://github.com/jrfiedler/xynn.

    Args:
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        device : string or torch.device

    """

    def __init__(
        self, cat_vc: Sequence[Dict], embedding_size: int = 10, device: Union[str, torch.device] = "cuda:0", **kwargs
    ):
        super().__init__()
        self._device = device
        self._isfit = False
        self.num_fields = 0
        self.output_size = 0
        self.lookup: Dict[Tuple[int, Any], int] = {}
        self.lookup_nan: Dict[int, int] = {}
        self.num_values = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._from_summary(cat_vc)
        self.cat_len = len(cat_vc)

    def _from_summary(self, uniques: List[Union[List, Tensor, np.ndarray]]):
        lookup = {}
        lookup_nan = {}
        num_values = 0
        for fieldnum, field in enumerate(uniques):
            for value in field:
                if (fieldnum, value) in lookup:
                    # extra defense against repeated values
                    continue
                lookup[(fieldnum, value)] = num_values
                num_values += 1

        self.num_fields = len(uniques)
        self.output_size = self.num_fields * self.embedding_size
        self.lookup = lookup
        self.lookup_nan = lookup_nan
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, self.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self._isfit = True

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.cat_len

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        if not self._isfit:
            raise RuntimeError("need to call `fit` or `from_summary` first")
        X = X["cat"]
        idxs: List[List[int]] = []
        for row in X:
            idxs.append([])
            for col, val in enumerate(row):
                val = val.item()
                idx = self.lookup[(col, val)]
                idxs[-1].append(idx)

        return self.embedding(torch.tensor(idxs, dtype=torch.int64, device=self._device))


class DefaultEmbedding(nn.Module):
    """DefaultEmbedding from https://github.com/jrfiedler/xynn.

    An embedding with a default value for each field. The default is returned for
    any field value not seen when the embedding was initialized (using `fit` or
    `from_summary`). For any value seen at initialization, a weighted average of
    that value's embedding and the default embedding is returned. The weights for
    the average are determined by the parameter `alpha`:

    weight = count / (count + alpha)
    final = embedding * weight + default * (1 - weight)

    Args:
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        alpha : int, optional
            controls the weighting of each embedding vector with the default;
            when `alpha`-many values are seen at initialization; the final
            vector is evenly weighted; the influence of the default is decreased
            with either higher counts or lower `alpha`; default is 20
        device : string or torch.device

    """

    def __init__(
        self,
        cat_vc: Sequence[Dict],
        embedding_size: int = 10,
        alpha: int = 20,
        device: Union[str, torch.device] = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self._isfit = False
        self._device = device
        self.num_fields = 0
        self.output_size = 0
        self.alpha = alpha
        self.lookup: Dict[Tuple[int, Any], Tuple[int, int]] = {}
        self.lookup_default: Dict[int, Tuple[int, int]] = {}
        self.num_values = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._from_summary(cat_vc)
        self.cat_len = len(cat_vc)

    def _from_summary(self, unique_counts: List[Dict[Any, int]]):
        lookup = {}
        lookup_default = {}
        num_values = 0
        for fieldnum, counts in enumerate(unique_counts):
            lookup_default[fieldnum] = (num_values, 0)
            num_values += 1
            for value, count in counts.items():
                lookup[(fieldnum, value)] = (num_values, count)
                num_values += 1

        self.num_fields = len(unique_counts)
        self.output_size = self.num_fields * self.embedding_size
        self.lookup = lookup
        self.lookup_default = lookup_default
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, self.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)

        self._isfit = True

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.cat_len

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor
        """
        if not self._isfit:
            raise RuntimeError("need to call `fit` or `from_summary` first")
        X = X["cat"]
        list_weights: List[List[List[float]]] = []
        idxs_primary: List[List[int]] = []
        idxs_default: List[List[int]] = []
        for row in X:
            list_weights.append([])
            idxs_primary.append([])
            idxs_default.append([])
            for col, val in enumerate(row):
                val = val.item()
                default = self.lookup_default[col]
                idx, count = self.lookup.get((col, val), default)
                list_weights[-1].append([count / (count + self.alpha)])
                idxs_primary[-1].append(idx)
                idxs_default[-1].append(default[0])
        tsr_weights = torch.tensor(list_weights, dtype=torch.float32, device=self._device)
        emb_primary = self.embedding(torch.tensor(idxs_primary, dtype=torch.int64, device=self._device))
        emb_default = self.embedding(torch.tensor(idxs_default, dtype=torch.int64, device=self._device))
        x = tsr_weights * emb_primary + (1 - tsr_weights) * emb_default
        return x


class LinearEmbedding(nn.Module):
    """An embedding for numeric fields from https://github.com/jrfiedler/xynn.

    There is one embedded vector for each field.
    The embedded vector for a value is that value times its field's vector.

    Args:
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        device : string or torch.device

    """

    def __init__(self, num_dims: int, embedding_size: int = 10, **kwargs):
        super().__init__()
        self._isfit = False
        self.num_fields = num_dims
        self.output_size = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._from_summary(self.num_fields)

    def _from_summary(self, num_fields: int):
        self.num_fields = num_fields
        self.output_size = num_fields * self.embedding_size
        self.embedding = nn.Embedding(num_fields, self.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self._isfit = True

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.num_fields

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cont"]
        if not self._isfit:
            raise RuntimeError("need to call `fit` or `from_summary` first")
        return self.embedding.weight * X.unsqueeze(dim=-1)


class DenseEmbedding(nn.Module):
    """An embedding for numeric fields, consisting of just a linear transformation with an activation from https://github.com/jrfiedler/xynn.

    Maps an input with shape n_rows * n_fields to an output with shape
    n_rows * 1 * embedding_size if one value passed for embedding_size or
    n_rows * embeddin_size[0] * embedding_size[1] if two values are passed

    Args:
        embedding_size : int, tuple of ints, or list of ints; optional
            size of each value's embedding vector; default is 10
        activation : subclass of torch.nn.Module, optional
            default is nn.LeakyReLU
        device : string or torch.device
    """

    def __init__(
        self,
        num_dims: int,
        embedding_size: Union[int, Tuple[int, ...], List[int]] = 10,
        activation: Type[nn.Module] = nn.LeakyReLU,
        **kwargs,
    ):
        super().__init__()

        if isinstance(embedding_size, int):
            embedding_size = (1, embedding_size)
        elif len(embedding_size) == 1:
            embedding_size = (1, embedding_size[0])
        self._isfit = False
        self.num_fields = num_dims
        self.output_size = 0
        self.embedding_w = None
        self.embedding_b = None
        self.dense_out_size = embedding_size
        self.embedding_size = embedding_size[-1]
        self.activation = activation()
        self._from_summary(self.num_fields)

    def _from_summary(self, num_fields: int):
        self.output_size = reduce(operator.mul, self.dense_out_size, 1)
        self.embedding_w = nn.Parameter(torch.zeros((num_fields, *self.dense_out_size)))
        self.embedding_b = nn.Parameter(torch.zeros(self.dense_out_size))
        nn.init.xavier_uniform_(self.embedding_w)
        self._isfit = True

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        return self.dense_out_size[0]

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cont"]
        if not self._isfit:
            raise RuntimeError("need to call `fit` or `from_summary` first")
        embedded = self.embedding_w.T.matmul(X.T.to(dtype=torch.float)).T + self.embedding_b
        embedded = self.activation(embedded.reshape((X.shape[0], -1)))
        return embedded.reshape((X.shape[0], *self.dense_out_size))


class TorchUniversalModel(nn.Module):
    """Mixed data model.

    Class for preparing input for DL model with mixed data.

    Args:
        loss: Callable torch loss with order of arguments (y_true, y_pred).
        task: Task object.
        torch_model: Torch model.
        n_out: Number of output dimensions.
        cont_embedder: Torch module for numeric data.
        cont_params: Dict with numeric model params.
        cat_embedder: Torch module for category data.
        cat_params: Dict with category model params.
        text_embedder: Torch module for text data.
        text_params: Dict with text model params.
        loss_on_logits: Calculate loss on logits or on predictions of model for classification tasks.
        bias: Array with last hidden linear layer bias.

    """

    def __init__(
        self,
        loss: Callable,
        task: Task,
        torch_model: nn.Module,
        n_out: int = 1,
        cont_embedder: Optional[Any] = None,
        cont_params: Optional[Dict] = None,
        cat_embedder: Optional[Any] = None,
        cat_params: Optional[Dict] = None,
        text_embedder: Optional[Any] = None,
        text_params: Optional[Dict] = None,
        loss_on_logits: bool = True,
        bias: Union[np.array, torch.Tensor] = None,
        **kwargs,
    ):
        super(TorchUniversalModel, self).__init__()
        self.n_out = n_out
        self.loss = loss
        self.task = task
        self.loss_on_logits = loss_on_logits

        self.cont_embedder = None
        self.cat_embedder = None
        self.text_embedder = None

        n_in = 0
        if cont_embedder is not None:
            self.cont_embedder = cont_embedder(**cont_params)
            n_in += self.cont_embedder.get_out_shape()
        if cat_embedder is not None:
            self.cat_embedder = cat_embedder(**cat_params)
            n_in += self.cat_embedder.get_out_shape()
        if text_embedder is not None:
            self.text_embedder = text_embedder(**text_params)
            n_in += self.text_embedder.get_out_shape()

        self.torch_model = (
            torch_model(
                **{
                    **kwargs,
                    **{
                        "n_in": n_in,
                        "n_out": n_out,
                        "loss": loss,
                        "task": task,
                    },
                }
            )
            if torch_model is not None
            else nn.Sequential(nn.Linear(n_in, n_out))
        )

        if bias is not None:
            try:
                last_layer = list(
                    filter(
                        lambda x: isinstance(x, nn.Linear) or isinstance(x, nn.Sequential),
                        list(self.torch_model.children()),
                    )
                )[-1]
                while isinstance(last_layer, nn.Sequential):
                    last_layer = list(
                        filter(lambda x: isinstance(x, nn.Linear) or isinstance(x, nn.Sequential), last_layer)
                    )[-1]
                bias = torch.Tensor(bias)
                last_layer.bias.data = bias
                shape = last_layer.weight.data.shape
                last_layer.weight.data = torch.zeros(shape[0], shape[1], requires_grad=True)
            except:
                logger.info3("Last linear layer not founded, so init_bias=False")

        self.сlump = Clump()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def get_logits(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-pass of model with embeddings."""
        outputs = []
        if self.cont_embedder is not None:
            outputs.append(self.cont_embedder(inp))

        if self.cat_embedder is not None:
            outputs.append(self.cat_embedder(inp))

        if self.text_embedder is not None:
            outputs.append(self.text_embedder(inp))

        if len(outputs) > 1:
            output = torch.cat(outputs, dim=1)
        else:
            output = outputs[0]

        logits = self.torch_model(output)
        return logits

    def get_preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Prediction from logits."""
        if self.task.name in ["binary", "multilabel"]:
            out = self.sig(self.сlump(logits))
        elif self.task.name == "multiclass":
            # cant find self.clump when predicting
            out = self.softmax(torch.clamp(logits, -50, 50))
        else:
            out = logits

        return out

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-pass with output loss."""
        x = self.get_logits(inp)
        if not self.loss_on_logits:
            x = self.get_preds_from_logits(x)

        loss = self.loss(inp["label"].view(inp["label"].shape[0], -1), x, inp.get("weight", None))
        return loss

    def predict(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prediction."""
        x = self.get_logits(inp)
        x = self.get_preds_from_logits(x)
        return x
