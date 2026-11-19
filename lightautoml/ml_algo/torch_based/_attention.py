"""Limit the size and lifetime of intermediate self-attention scores."""

from inspect import signature

import torch

from torch.utils.checkpoint import checkpoint


_SCORE_BLOCK_BYTES = 32 * 1024 ** 2
try:
    _HAS_NON_REENTRANT_CHECKPOINT = "use_reentrant" in signature(checkpoint).parameters
except (TypeError, ValueError):
    # The documentation build imports this module with a mocked torch.
    _HAS_NON_REENTRANT_CHECKPOINT = False


def attention_by_query(attend, query, key, *args, num_heads):
    """Split independent query fields while keeping the softmax over all keys."""
    batch, fields = query.shape[:2]
    # A score block has shape (batch, heads, query_fields, key_fields).
    # The math backend may compute the scores in float32 for half inputs.
    bytes_per_query = batch * num_heads * key.size(1) * max(query.element_size(), 4)
    block_size = max(1, _SCORE_BLOCK_BYTES // max(bytes_per_query, 1))
    if fields <= block_size:
        return attend(query, key, *args)

    blocks = []
    for block in query.split(block_size, dim=1):
        if torch.is_grad_enabled() and _HAS_NON_REENTRANT_CHECKPOINT:
            # Recompute each block's scores in backward, with the same dropout mask.
            out = checkpoint(attend, block, key, *args, use_reentrant=False)
        else:
            out = attend(block, key, *args)
        blocks.append(out)
    return torch.cat(blocks, dim=1)
