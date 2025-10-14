"""Custom BatchSampler for TabM with share_training_batches support."""

import torch
from torch.utils.data import Sampler
from typing import Union


class TabMBatchSampler(Sampler):
    """Custom BatchSampler for TabM with share_training_batches support.

    Note: self.shuffle is not ignored and always performs shuffle if share_training_batches is True.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        k: int,
        share_training_batches: bool = True,
        device: Union[str, torch.device] = "cuda:0",
        shuffle=True,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k = k
        self.share_training_batches = share_training_batches
        self.device = device
        self.shuffle = shuffle

        if self.share_training_batches:
            # Create one standard batch sequence.
            if self.shuffle:
                batches_tensor = torch.randperm(dataset_size, device=device).split(batch_size)
            else:
                batches_tensor = torch.arange(dataset_size, device=device).split(batch_size)
            self.batches = [batch.cpu().numpy() for batch in batches_tensor]
        else:
            # Create k independent batch sequences.
            batches_tensor = torch.rand((dataset_size, k), device=device).argsort(dim=0).split(batch_size, dim=0)
            self.batches = [batch.cpu().numpy() for batch in batches_tensor]

    def __iter__(self):
        """Batch iterator."""
        for batch in self.batches:
            yield batch

    def __len__(self):
        """Number of batches."""
        if self.share_training_batches:
            return len(self.batches)
        else:
            return len(self.batches) * self.k
