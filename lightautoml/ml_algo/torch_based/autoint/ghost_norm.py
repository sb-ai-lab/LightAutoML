"""Module for Ghost Batch Norm and variations.

Ghost Batch Norm: https://arxiv.org/pdf/1705.08741.pdf

"""

from math import ceil
from typing import Union

import torch
from torch import Tensor
from torch import nn


class GhostNorm(nn.Module):
    """Ghost Normalization.

    https://arxiv.org/pdf/1705.08741.pdf

    Args:
        inner_norm : torch.nn.Module (initialiezd)
            examples: `nn.BatchNorm1d`, `nn.LayerNorm`
        virtual_batch_size : int
        device : string or torch.device, optional
            default is "cpu"
    """

    def __init__(
        self,
        inner_norm: nn.Module,
        virtual_batch_size: int,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.inner_norm = inner_norm
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Transform the input tensor.

        Args:
            x : torch.Tensor

        Returns:
            torch.Tensor

        """
        chunk_size = int(ceil(x.shape[0] / self.virtual_batch_size))
        chunk_norm = [self.inner_norm(chunk) for chunk in x.chunk(chunk_size, dim=0)]
        return torch.cat(chunk_norm, dim=0)


class GhostBatchNorm(GhostNorm):
    """Ghost Normalization, using BatchNorm1d as inner normalization.

    https://arxiv.org/pdf/1705.08741.pdf

    Args:
        num_features : int
        virtual_batch_size : int, optional
            default is 64
        momentum : float, optional
            default is 0.1
        device : string or torch.device, optional
            default is "cpu"
    """

    def __init__(
        self,
        num_features: int,
        virtual_batch_size: int = 64,
        momentum: float = 0.1,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            inner_norm=nn.BatchNorm1d(num_features, momentum=momentum),
            virtual_batch_size=virtual_batch_size,
        )
