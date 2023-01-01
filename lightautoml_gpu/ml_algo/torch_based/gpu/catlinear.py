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

class CatLogisticRegression(CatLinear):
    """Realisation of torch-based logistic regression (GPU version)."""

    def __init__(
        self, numeric_size: int, embed_sizes: Sequence[int] = (), output_size: int = 1
    ):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        numbers: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
    ):
        """Forward-pass. Sigmoid func at the end of linear layer.

        Args:
            numbers: Input numeric features.
            categories: Input categorical features.

        """
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.sigmoid(x)

        return x


class CatRegression(CatLinear):
    """Realisation of torch-based linear regreession (GPU version)."""

    def __init__(
        self, numeric_size: int, embed_sizes: Sequence[int] = (), output_size: int = 1
    ):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)


class CatMulticlass(CatLinear):
    """Realisation of multi-class linear classifier (GPU version)."""

    def __init__(
        self, numeric_size: int, embed_sizes: Sequence[int] = (), output_size: int = 1
    ):
        super().__init__(numeric_size, embed_sizes=embed_sizes, output_size=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        numbers: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
    ):
        x = super().forward(numbers, categories)
        x = torch.clamp(x, -50, 50)
        x = self.softmax(x)

        return x
