"""Neural Net modules for different data types."""

import logging

from typing import Any, List, Tuple, Type, Literal
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
from functools import reduce
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import operator
import numpy as np
from ..transformers.categorical import _check_bins
import warnings

try:
    from transformers import AutoModel
except:

    warnings.warn("'transformers' - package isn't installed")

from .dl_transformers import pooling_by_name


logger = logging.getLogger(__name__)


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
            raise ValueError(f"pooling - {pooling} - not in the list of available types {self._poolers}")

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


class BasicCatEmbedding(nn.Module):
    """A basic embedding that creates an embedded vector for each field value from https://github.com/jrfiedler/xynn.

    The same as CatEmbedder, but without dropout, and it can be presented as a sequence.

    Args:
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        device : string or torch.device
        flatten_output: if flatten output or not.

    """

    def __init__(
        self,
        cat_dims: Sequence[int],
        embedding_size: int = 10,
        device: Union[str, torch.device] = "cuda:0",
        flatten_output: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.flatten_output = flatten_output
        self._device = device
        self.num_fields = 0
        self.output_size = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._from_summary(cat_dims)
        self.cat_len = len(cat_dims)

    def _from_summary(self, cat_dims: Sequence[int]):
        num_values = 0

        self.emb_layers = nn.ModuleList([nn.Embedding(int(x), self.embedding_size) for x in cat_dims])
        self.num_fields = len(cat_dims)
        self.output_size = self.num_fields * self.embedding_size
        self.num_values = num_values
        for emb in self.emb_layers:
            nn.init.xavier_uniform_(emb.weight)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.cat_len * self.embedding_size
        else:
            return self.cat_len

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cat"]
        x = torch.stack(
            [emb_layer(X[:, i]) for i, emb_layer in enumerate(self.emb_layers)],
            dim=1,
        )
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


class WeightedCatEmbedding(nn.Module):
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
        flatten_output: if flatten output or not.

    """

    def __init__(
        self,
        cat_dims: Sequence[int],
        cat_vc: Sequence[Dict],
        embedding_size: int = 10,
        alpha: int = 20,
        device: Union[str, torch.device] = "cuda:0",
        flatten_output: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.flatten_output = flatten_output
        self._device = device
        self.num_fields = 0
        self.output_size = 0
        self.alpha = alpha
        self.lookup: Dict[Tuple[int, Any], Tuple[int, int]] = {}
        self.lookup_default: Dict[int, Tuple[int, int]] = {}
        self.num_values = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._from_summary(cat_vc, cat_dims)
        self.cat_len = len(cat_vc)
        self.cat_dims = cat_dims

    def _from_summary(self, unique_counts: List[Dict[Any, int]], cat_dims: Sequence[int]):
        self.emb_layers = nn.ModuleList([nn.Embedding(int(x), self.embedding_size) for x in cat_dims])
        self.def_layers = nn.ModuleList([nn.Embedding(1, self.embedding_size) for _ in cat_dims])
        weights_list = []
        for fieldnum, counts in enumerate(unique_counts):
            weights = []
            for i, vc in enumerate(sorted(counts.items())):
                value, count = vc
                if i == 0 and value != 0.0:
                    weights.append([0])
                weights.append([count / (count + self.alpha)])
            weights_list.append(weights)
        self.w_emb_layers = nn.ModuleList(
            [nn.Embedding.from_pretrained(torch.tensor(x, dtype=torch.float32)) for x in weights_list]
        )
        self.num_fields = len(unique_counts)
        self.output_size = self.num_fields * self.embedding_size
        for emb in self.emb_layers:
            nn.init.xavier_uniform_(emb.weight)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.cat_len * self.embedding_size
        else:
            return self.cat_len

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor
        """
        X = X["cat"]
        emb_primary = torch.stack(
            [emb_layer(X[:, i]) for i, emb_layer in enumerate(self.emb_layers)],
            dim=1,
        )
        tsr_weights = torch.stack(
            [emb_layer(X[:, i]) for i, emb_layer in enumerate(self.w_emb_layers)],
            dim=1,
        )

        emb_default = torch.stack(
            [
                emb_layer(torch.tensor([0] * len(X[:, i]), device=self._device))
                for i, emb_layer in enumerate(self.def_layers)
            ],
            dim=1,
        )

        x = tsr_weights * emb_primary + (1 - tsr_weights) * emb_default
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


class LinearEmbedding(nn.Module):
    """An embedding for numeric fields from https://github.com/jrfiedler/xynn.

    There is one embedded vector for each field.
    The embedded vector for a value is that value times its field's vector.

    Args:
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        device : string or torch.device
        flatten_output: if flatten output or not.

    """

    def __init__(self, num_dims: int, embedding_size: int = 10, flatten_output: bool = False, **kwargs):
        super().__init__()
        self.flatten_output = flatten_output
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

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.num_fields * self.embedding_size
        else:
            return self.num_fields

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cont"]
        x = self.embedding.weight * X.unsqueeze(dim=-1)
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


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
        flatten_output: if flatten output or not.
    """

    def __init__(
        self,
        num_dims: int,
        embedding_size: Union[int, Tuple[int, ...], List[int]] = 10,
        activation: Type[nn.Module] = nn.LeakyReLU,
        flatten_output: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.flatten_output = flatten_output
        if isinstance(embedding_size, int):
            embedding_size = (1, embedding_size)
        elif len(embedding_size) == 1:
            embedding_size = (1, embedding_size[0])
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

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.output_size
        else:
            return self.dense_out_size[0]

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cont"]
        embedded = self.embedding_w.T.matmul(X.T.float()).T + self.embedding_b
        embedded = self.activation(embedded.reshape((X.shape[0], -1)))
        x = embedded.reshape((X.shape[0], *self.dense_out_size))
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


class DenseEmbeddingFlat(DenseEmbedding):
    """Flatten version of DenseEmbedding."""

    def __init__(self, *args, **kwargs):
        super(DenseEmbeddingFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})


class LinearEmbeddingFlat(LinearEmbedding):
    """Flatten version of LinearEmbedding."""

    def __init__(self, *args, **kwargs):
        super(LinearEmbeddingFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})


class WeightedCatEmbeddingFlat(WeightedCatEmbedding):
    """Flatten version of WeightedCatEmbedding."""

    def __init__(self, *args, **kwargs):
        super(WeightedCatEmbeddingFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})


class BasicCatEmbeddingFlat(BasicCatEmbedding):
    """Flatten version of BasicCatEmbedding."""

    def __init__(self, *args, **kwargs):
        super(BasicCatEmbeddingFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})


class NLinearMemoryEfficient(nn.Module):
    """Linear multi-dim embedding from https://github.com/yandex-research/tabular-dl-num-embeddings/tree/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8.

    Args:
        n : num of features.
        d_in: input size.
        d_out: output size.
    """

    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        """Forward-pass."""
        return torch.stack([layer(x[:, i]) for i, layer in enumerate(self.layers)], 1)


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings.

    Original sources:
    - 'On Embeddings for Numerical Features in Tabular Deep Learning' Gorishniy et al. (2022)
    (https://arxiv.org/pdf/2203.05556, https://github.com/yandex-research/rtdl-num-embeddings).

    In other words,
    each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(
                "_NLinear supports only inputs with exactly one batch dimension,"
                " so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING)."
            )
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class Periodic(nn.Module):
    """Periodic positional embedding for numeric features from https://github.com/yandex-research/tabular-dl-num-embeddings/tree/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8.

    Args:
        n_features: num of numeric features
        emb_size: output size will be 2*emb_size
        sigma: weights will be initialized with N(0,sigma)
        flatten_output: if flatten output or not.
    """

    def __init__(
        self, n_features: int, emb_size: int = 64, sigma: float = 0.05, flatten_output: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.emb_size = emb_size
        coefficients = torch.normal(0.0, sigma, (n_features, emb_size))
        self.coefficients = nn.Parameter(coefficients)
        self.flatten_output = flatten_output

    @staticmethod
    def _cos_sin(x: Tensor) -> Tensor:
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.emb_size * 2 * self.n_features
        else:
            return self.n_features

    def forward(self, x: Tensor) -> Tensor:
        """Forward-pass."""
        x = self._cos_sin(2 * np.pi * self.coefficients[None] * x[..., None])
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


class PLREmbedding(nn.Module):
    """ReLU ◦ Linear ◦ Periodic embedding for numeric features from https://arxiv.org/pdf/2203.05556.pdf.

    Args:
        num_dims: int
        emb_size: int
        sigma: float
        flatten_output : bool
    """

    def __init__(
        self,
        num_dims: int,
        embedding_size: Union[int, Tuple[int, ...], List[int]] = 64,
        emb_size_periodic: int = 64,
        sigma_periodic: float = 0.05,
        flatten_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.embedding_size = embedding_size
        self.layers: List[nn.Module] = []
        self.layers.append(Periodic(num_dims, emb_size_periodic, sigma_periodic))
        self.layers.append(NLinearMemoryEfficient(num_dims, 2 * emb_size_periodic, embedding_size))
        self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)
        self.flatten_output = flatten_output

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.num_dims * self.embedding_size
        else:
            return self.num_dims

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cont"]
        x = self.layers(X)
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


class PLREmbeddingFlat(PLREmbedding):
    """Flatten version of PLREmbedding."""

    def __init__(self, *args, **kwargs):
        super(PLREmbeddingFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})


class SoftEmbedding(torch.nn.Module):
    """Soft-one hot encoding embedding technique, from https://arxiv.org/pdf/1708.00065.pdf.

    In a nutshell, it represents a continuous feature as a weighted average of embeddings

    Args:
        num_embeddings: Number of embeddings to use (cardinality of the embedding table).
        embeddings_dim: The dimension of the vector space for projecting the scalar value.
        embeddings_init_std: The standard deviation factor for normal initialization of the
            embedding matrix weights.
        emb_initializer: Dict where keys are feature names and values are callable to initialize
            embedding tables
    """

    def __init__(self, num_dims, embedding_size=10, flatten_output: bool = False, **kwargs) -> None:
        super(SoftEmbedding, self).__init__()
        self.embedding_table = torch.nn.Embedding(num_dims, embedding_size)
        nn.init.xavier_uniform_(self.embedding_table.weight)

        self.projection_layer = torch.nn.Linear(1, num_dims, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.emb_size = embedding_size
        self.num_dims = num_dims
        self.flatten_output = flatten_output

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.num_dims * self.emb_size
        else:
            return self.num_dims

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        X = X["cont"]
        input_numeric = X.unsqueeze(-1)
        weights = self.softmax(self.projection_layer(input_numeric))
        x = (weights.unsqueeze(-1) * self.embedding_table.weight).sum(-2)
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x


class SoftEmbeddingFlat(SoftEmbedding):
    """Flatten version of SoftEmbedding."""

    def __init__(self, *args, **kwargs):
        super(SoftEmbeddingFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})


def _check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(f"The input must have at least one dimension, however: {x.ndim=}")
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            "The last dimension of the input was expected to be" f" {expected_n_features}, however, {x.shape[-1]=}"
        )


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m.get_output_shape()
    torch.Size([3, 4])
    >>> m(x).shape
    torch.Size([2, 3, 4])

    Args:
        n_features: the number of continuous features.
        d_embedding: the embedding size.
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Init parameters."""
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        return self.weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        return torch.addcmul(self.bias, self.weight, x[..., None])


class _PiecewiseLinearEncodingImpl(nn.Module):
    """Piecewise-linear encoding.

    Original sources:
    - 'On Embeddings for Numerical Features in Tabular Deep Learning' Gorishniy et al. (2022)
    (https://arxiv.org/pdf/2203.05556, https://github.com/yandex-research/rtdl-num-embeddings).


    NOTE: THIS CLASS SHOULD NOT BE USED DIRECTLY.
    In particular, this class does *not* add any positional information
    to feature encodings. Thus, for Transformer-like models,
    `PiecewiseLinearEmbeddings` is the only valid option.

    Note:
        This is the *encoding* module, not the *embedding* module,
        so it only implements Equation 1 (Figure 1) from the paper,
        and does not have trainable parameters.

    **Shape**

    * Input: ``(*, n_features)``
    * Output: ``(*, n_features, max_n_bins)``,
      where ``max_n_bins`` is the maximum number of bins over all features:
      ``max_n_bins = max(len(b) - 1 for b in bins)``.

    To understand the output structure,
    consider a feature with the number of bins ``n_bins``.
    Formally, its piecewise-linear encoding is a vector of the size ``n_bins``
    that looks as follows::

        x_ple = [1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]

    However, this class will instead produce a vector of the size ``max_n_bins``::

        x_ple_actual = [*x_ple[:-1], *zeros(max_n_bins - n_bins), x_ple[-1]]

    In other words:

    * The last encoding component is **always** located in the end,
      even if ``n_bins == 1`` (i.e. even if it is the only component).
    * The leading ``n_bins - 1`` components are located in the beginning.
    * Everything in-between is always set to zeros (like "padding", but in the middle).

    This implementation is *significantly* faster than the original one.
    It relies on two key observations:

    * The piecewise-linear encoding is just
      a non-trainable linear transformation followed by a clamp-based activation.
      Pseudocode: `PiecewiseLinearEncoding(x) = Activation(Linear(x))`.
      The parameters of the linear transformation are defined by the bin edges.
    * Aligning the *last* encoding channel across all features
      allows applying the aforementioned activation simultaneously to all features
      without the loop over features.

    Args:
        bins: the bins computed by `compute_bins`.
    """

    weight: Tensor
    """The weight of the linear transformation mentioned in the class docstring."""

    bias: Tensor
    """The bias of the linear transformation mentioned in the class docstring."""

    single_bin_mask: Optional[Tensor]
    """The indicators of the features with only one bin."""

    mask: Optional[Tensor]
    """The indicators of the "valid" (i.e. "non-padding") part of the encoding."""

    def __init__(self, bins: List[torch.Tensor]) -> None:
        assert len(bins) > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        self.register_buffer("weight", torch.zeros(n_features, max_n_bins))
        self.register_buffer("bias", torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer("single_bin_mask", single_bin_mask if single_bin_mask.any() else None)

        self.register_buffer(
            "mask",
            # The mask is needed if features have different number of bins.
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            # Unused components (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                        if len(x) < 2
                        else [
                            # The number of bins for this feature, minus 1:
                            torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                            # Unused components (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    # x is a tensor containing the bin bounds for a given feature.
                    for x in bins
                ]
            ),
        )

        for i, bin_edges in enumerate(bins):
            # Formally, the piecewise-linear encoding of one feature looks as follows:
            # `[1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]`
            # The linear transformation based on the weight and bias defined below
            # implements the expression in the middle before the clipping to [0, 1].
            # Note that the actual encoding layout produced by this class
            # is slightly different. See the docstring of this class for details.
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # The last encoding component:
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            # The leading encoding components:
            self.weight[i, : n_bins[i] - 1] = w[:-1]
            self.bias[i, : n_bins[i] - 1] = b[:-1]
            # All in-between components will always be zeros,
            # because the weight and bias are initialized with zeros.

    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]

    def forward(self, x: torch.Tensor) -> Tensor:
        """Do the forward pass."""
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.shape[-1] > 1:
            x = torch.cat(
                [
                    x[..., :1].clamp_max(1.0),
                    x[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            # For features with only one bin,
                            # the whole "piecewise-linear" encoding effectively behaves
                            # like mix-max scaling
                            # (assuming that the edges of the single bin
                            #  are the minimum and maximum feature values).
                            self.single_bin_mask[..., None],
                            x[..., -1:],
                            x[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        return x


class PiecewiseLinearEmbeddings(nn.Module):
    """Piecewise-linear embeddings actually is just a Piecewise-linear encodings with simple differentiable layers on top.

    Original sources:
    1) 'On Embeddings for Numerical Features in Tabular Deep Learning' Gorishniy et al. (2022):
    https://arxiv.org/pdf/2203.05556, https://github.com/yandex-research/rtdl-num-embeddings
    2) 'TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling' Gorishniy et al. (2025):
    https://arxiv.org/abs/2410.24210v3.

    The intuition behind PiecewiseLinearEmbeddings is that each bin receives its own trainable embedding,
    and the feature embedding is the aggregation of its bin embeddings with the aggregation weights provided
    by the piecewise-linear encoding.

    PiecewiseLinearEncoding represents a fixed (non-trainable) transformation, fully defined by the provided bin
    boundaries. Its output is the concatenation of the piecewise-linear representations of all features, i.e.
    this representation has the shape (d_encoding,), where d_encoding equals the total number of bins across all features.
    Because of that, PiecewiseLinearEncoding can be used only with MLP-like models, but not with Transformer-like models.

    In practice, if there is enough data to train additional weights, PiecewiseLinearEmbeddings may be a better starting
    point even for MLP-like models. In particular, PiecewiseLinearEmbeddings makes it possible to set a large number of
    bins, but a small embedding size.

    Input: ``(batch_size, n_features)``
    Output: ``(batch_size, n_features, embedding_size)`` or ``(batch_size, n_features * embedding_size)``

    Args:
        num_dims: Number of numeric features.
        bins: the quantile-based or target-aware bins computed by `compute_bins`.
        embedding_size: the embedding size.
        activation: if True, the ReLU activation is additionally applied in the end.
        version: the preset for various implementation details, such as
            parametrization and initialization. "A" is the original version used in the paper.
            "B" is the version used in the different paper about the TabM model. "A" denotes (ReLU) ◦ Linear ◦ PLE in terms from paper.
            "B" adds LinearEmbedding to "A". In version "B", the whole embedding behaves like a linear embedding at the start,
            and the piecewise-linear component is incrementally learnt during training.
        flatten_output: bool: output shape is (batch_size, n_features, embedding_size)`` or ``(batch_size, n_features * embedding_size).
    """

    def __init__(
        self,
        num_dims: int,
        embedding_size: int,
        bins: List[Tensor] = None,
        activation: bool = False,
        version: Literal[None, "A", "B"] = "B",
        flatten_output: bool = False,
        **kwargs,
    ) -> None:
        if embedding_size <= 0:
            raise ValueError(f"embedding_size must be a positive integer, however: {embedding_size=}")

        super().__init__()

        _check_bins(bins)
        self.bins = bins
        n_features = len(bins)
        is_version_B = version == "B"

        self.linear0 = LinearEmbeddings(n_features, embedding_size) if is_version_B else None
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            embedding_size,
            bias=not is_version_B,  # For the version "B", the bias is already presented in self.linear0.
        )
        if is_version_B:
            # Because of the following line, at initialization,
            # the whole embedding behaves like a linear embedding.
            # The piecewise-linear component is incrementally learnt during training.
            nn.init.zeros_(self.linear.weight)
        self.activation = nn.ReLU() if activation else None

        self.num_dims = num_dims
        self.flatten_output = flatten_output
        self.embedding_size = embedding_size

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.num_dims * self.embedding_size
        else:
            return self.num_dims

    def forward(self, X: Dict) -> Tensor:
        """Forward pass."""
        x = X["cont"]

        if x.ndim != 2:
            raise ValueError("For now, only inputs with exactly one batch dimension are supported.")

        x_linear = None if self.linear0 is None else self.linear0(x)

        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)
        if self.activation is not None:
            x_ple = self.activation(x_ple)
        output = x_ple if x_linear is None else x_linear + x_ple

        if self.flatten_output:
            return output.contiguous().view(output.shape[0], -1)
        return output


class PiecewiseLinearEmbeddingsFlat(PiecewiseLinearEmbeddings):
    """Flatten version of PiecewiseLinearEmbeddings."""

    def __init__(self, *args, **kwargs):
        super(PiecewiseLinearEmbeddingsFlat, self).__init__(*args, **{**kwargs, **{"flatten_output": True}})
