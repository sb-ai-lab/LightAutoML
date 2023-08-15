"""PyTorch modules for the AutoInt model."""
# Paper: https://arxiv.org/pdf/1810.11921v2.pdf
# Official implementation: https://github.com/DeepGraphLearning/RecommenderSystems

from collections import namedtuple
from typing import Optional, Type, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F


EmbeddingInfo = namedtuple("EmbeddingInfo", ["num_fields", "output_size"])
UniformEmbeddingInfo = namedtuple("EmbeddingInfo", ["num_fields", "embedding_size", "output_size"])



class LeakyGate(nn.Module):
    """LeakyGate from https://github.com/jrfiedler/xynn.

    This performs an element-wise linear transformation followed by a chosen
    activation; the default activation is nn.LeakyReLU. Fields may be
    represented by individual values or vectors of values (i.e., embedded).

    Input needs to be shaped like (num_rows, num_fields) or
    (num_rows, num_fields, embedding_size)

    Args:
        input_size: input_size.
        bias: if to use bias.
        activation: activation function.
        device: device.
    """

    def __init__(
        self,
        input_size: int,
        bias: bool = True,
        activation: Type[nn.Module] = nn.LeakyReLU,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.normal(mean=0, std=1.0, size=(1, input_size)))
        self.bias = nn.Parameter(torch.zeros(size=(1, input_size)), requires_grad=bias)
        self.activation = activation()
        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        """Transform the input tensor.

        Args:
            X : torch.Tensor

        Returns:
            torch.Tensor
        """
        out = X
        if len(X.shape) > 2:
            out = out.reshape((X.shape[0], -1))
        out = out * self.weight + self.bias
        if len(X.shape) > 2:
            out = out.reshape(X.shape)
        out = self.activation(out)
        return out


def _initialized_tensor(*sizes):
    weight = nn.Parameter(torch.Tensor(*sizes))
    nn.init.kaiming_uniform_(weight)
    return weight


class AttnInteractionLayer(nn.Module):
    """The attention interaction layer for the AutoInt model.

    Paper for the original AutoInt model: https://arxiv.org/pdf/1810.11921v2.pdf

    Args:
        field_input_size : int
            original embedding size for each field
        field_output_size : int, optional
            embedding size after transformation; default is 8
        num_heads : int, optional
            number of attention heads; default is 2
        activation : subclass of torch.nn.Module or None, optional
            applied to the W tensors; default is None
        use_residual : bool, optional
            default is True
        dropout : float, optional
            default is 0.1
        normalize : bool, optional
            default is True
        ghost_batch_size : int or None, optional
            only used if `use_bn` is True; size of batch in "ghost batch norm";
            if None, normal batch norm is used; defualt is None
        device : string or torch.device, optional
            default is "cpu"

    """

    def __init__(
        self,
        field_input_size: int,
        field_output_size: int = 8,
        num_heads: int = 2,
        activation: Optional[Type[nn.Module]] = None,
        use_residual: bool = True,
        dropout: float = 0.1,
        normalize: bool = True,
        ghost_batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.use_residual = use_residual

        self.W_q = _initialized_tensor(field_input_size, field_output_size, num_heads)
        self.W_k = _initialized_tensor(field_input_size, field_output_size, num_heads)
        self.W_v = _initialized_tensor(field_input_size, field_output_size, num_heads)

        if use_residual:
            self.W_r = _initialized_tensor(field_input_size, field_output_size * num_heads)
        else:
            self.W_r = None

        if activation:
            self.w_act = activation()
        else:
            self.w_act = nn.Identity()

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if normalize:
            self.layer_norm = nn.LayerNorm(field_output_size * num_heads)
        else:
            self.layer_norm = nn.Identity()

        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Transform the input tensor with attention interaction.

        Args:
            x : torch.Tensor
                3-d tensor; for example, embedded numeric and/or categorical values,
                or the output of a previous attention interaction layer

        Returns:
            torch.Tensor

        """
        # R : # rows
        # F, D : # fields
        # I : field embedding size in
        # O : field embedding size out
        # H : # heads
        num_rows, num_fields, _ = x.shape  # R, F, I

        # (R, F, I) * (I, O, H) -> (R, F, O, H)
        qrys = torch.tensordot(x, self.w_act(self.W_q), dims=([-1], [0]))
        keys = torch.tensordot(x, self.w_act(self.W_k), dims=([-1], [0]))
        vals = torch.tensordot(x, self.w_act(self.W_v), dims=([-1], [0]))
        if self.use_residual:
            rsdl = torch.tensordot(x, self.w_act(self.W_r), dims=([-1], [0]))

        product = torch.einsum("rdoh,rfoh->rdfh", qrys, keys)  # (R, F, F, H)

        alpha = F.softmax(product, dim=2)  # (R, F, F, H)
        alpha = self.dropout(alpha)

        # (R, F, F, H) * (R, F, O, H) -> (R, F, O, H)
        out = torch.einsum("rfdh,rfoh->rfoh", alpha, vals)
        out = out.reshape((num_rows, num_fields, -1))  # (R, F, O * H)
        if self.use_residual:
            out = out + rsdl  # (R, F, O * H)
        out = F.leaky_relu(out)
        out = self.layer_norm(out)

        return out


class AttnInteractionBlock(nn.Module):
    """A collection of AttnInteractionLayers, followed by an optional "leaky gate" and then a linear layer.

    This block is originally for the AutoInt model.

    Code from: https://github.com/jrfiedler/xynn

    Args:
        field_input_size : int
            original embedding size for each field
        field_output_size : int, optional
            embedding size after transformation; default is 8
        num_layers : int, optional
            number of attention layers; default is 3
        num_heads : int, optional
            number of attention heads per layer; default is 2
        activation : subclass of torch.nn.Module or None, optional
            applied to the W tensors; default is None
        use_residual : bool, optional
            default is True
        dropout : float, optional
            default is 0.0
        normalize : bool, optional
            default is True
        ghost_batch_size : int or None, optional
            only used if `use_bn` is True; size of batch in "ghost batch norm";
            if None, normal batch norm is used; defualt is None
        device : string or torch.device, optional
            default is "cpu"
    """

    def __init__(
        self,
        field_input_size: int,
        field_output_size: int = 8,
        num_layers: int = 3,
        num_heads: int = 2,
        activation: Optional[Type[nn.Module]] = None,
        use_residual: bool = True,
        dropout: float = 0.1,
        normalize: bool = True,
        ghost_batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                AttnInteractionLayer(
                    field_input_size,
                    field_output_size,
                    num_heads,
                    activation,
                    use_residual,
                    dropout,
                    normalize,
                    ghost_batch_size,
                    device,
                )
            )
            field_input_size = field_output_size * num_heads

        self.layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Transform the input tensor.

        Args:
            x : torch.Tensor
                3-d tensor, usually embedded numeric and/or categorical values

        Returns:
            torch.Tensor
        """
        out = self.layers(x)
        return out
