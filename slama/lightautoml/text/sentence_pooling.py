"""Different Pooling strategies for sequence data."""


import torch
import torch.nn as nn


class SequenceAbstractPooler(nn.Module):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SequenceClsPooler(SequenceAbstractPooler):
    """CLS token pooling."""

    def __init__(self):
        super(SequenceClsPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        return x[..., 0, :]


class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~x_mask, -float("inf"))
        values, _ = torch.max(x, dim=-2)
        return values


class SequenceSumPooler(SequenceAbstractPooler):
    """Sum value pooling."""

    def __init__(self):
        super(SequenceSumPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~x_mask, 0)
        values = torch.sum(x, dim=-2)
        return values


class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~x_mask, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active.data
        # values = torch.mean(x, dim=-2)
        return values


class SequenceIndentityPooler(SequenceAbstractPooler):
    """Identity pooling."""

    def __init__(self):
        super(SequenceIndentityPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        return x
