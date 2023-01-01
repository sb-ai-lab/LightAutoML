"""Linear models based on Torch library."""

import logging
from typing import Optional, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CatLinear(nn.Module):
    """Simple linear model to handle numeric and categorical features (GPU version)."""

    def __init__(
        self,
        numeric_size: int = 0,
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
    ):
        """
        Args:
            numeric_size: Number of numeric features.
            embed_sizes: Embedding sizes.
            output_size: Size of output layer.

        """
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(output_size).cuda())

        # add numeric if it is defined
        self.linear = None
        if numeric_size > 0:
            self.linear = nn.Linear(
                in_features=numeric_size, out_features=output_size, bias=False
            ).cuda()
            nn.init.zeros_(self.linear.weight)

        # add categories if it is defined
        self.cat_params = None
        if len(embed_sizes) > 0:
            self.cat_params = nn.Parameter(
                torch.zeros(embed_sizes.sum(), output_size).cuda()
            )
            self.register_buffer(
                "embed_idx",
                torch.LongTensor(embed_sizes).cumsum(dim=0)
                - torch.LongTensor(embed_sizes),
            )

    def forward(
        self,
        numbers: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
    ):
        """Forward-pass.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        """
        x = self.bias

        if self.linear is not None:
            x = x + self.linear(numbers)

        if self.cat_params is not None:
            x = x + self.cat_params[categories + self.embed_idx].sum(dim=1)

        return x
