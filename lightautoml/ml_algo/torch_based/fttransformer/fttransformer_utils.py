"""Feedforward and Attention blocks for FTTransformer (https://arxiv.org/abs/2106.11959v2) from https://github.com/lucidrains/tab-transformer-pytorch/tree/main."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._attention import attention_by_query


class GEGLU(nn.Module):
    """GEGLU activation for Attention block."""

    def forward(self, x):
        """Forward pass."""
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.0):
    """Feedforward for Transformer block."""
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    """Attention Block.

    Args:
            dim: Embeddings dimension.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            dropout: Post-Attention dropout.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _weights(self, q, k):
        return torch.softmax(torch.matmul(q * self.scale, k.transpose(-2, -1)), dim=-1)

    def _attend(self, q, k, v):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.dropout.training else 0.0)
        else:
            out = torch.matmul(self.dropout(self._weights(q, k)), v)
        return out.transpose(1, 2)

    def forward(self, x, return_attn: bool = True):
        """Transform the input tensor with attention.

        Args:
            x : torch.Tensor
                3-d tensor; for example, embedded numeric and/or categorical values,
                or the output of a previous attention layer.
            return_attn: Return the pre-dropout attention weights alongside the output.

        Returns:
            Projected output and the attention weights, or an empty tensor if omitted.

        """
        batch_size, seq_len, dim = x.shape
        h = self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, qkv.size(-1) // 3, dim=-1)

        q = q.view(batch_size, seq_len, h, -1)
        k = k.view(batch_size, seq_len, h, -1)
        v = v.view(batch_size, seq_len, h, -1)
        if torch.jit.is_scripting() or return_attn:
            weights = self._weights(q.transpose(1, 2), k.transpose(1, 2))
            out = torch.matmul(self.dropout(weights), v.transpose(1, 2)).transpose(1, 2)
        else:
            out = attention_by_query(self._attend, q, k, v, num_heads=h)
        return self.to_out(out.reshape(batch_size, seq_len, -1)), weights if return_attn else x.new_empty((0,))


# transformer
class Transformer(nn.Module):
    """Transformer Block.

    Args:
            dim: Embeddings dimension.
            depth: Number of Attention Blocks.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            attn_dropout: Post-Attention dropout.
            ff_dropout: Feed-Forward Dropout.
    """

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x):
        """Transform the input embeddings tensor with Transformer module.

        Args:
            x : torch.Tensor
                3-d tensor; embedded numeric and/or categorical values,
                or the output of a previous Transformer layer.

        Returns:
            torch.Tensor

        """
        for attn, ff in self.layers:
            attn_out, _ = attn(x, return_attn=False)
            x = attn_out + x
            x = ff(x) + x

        return x
