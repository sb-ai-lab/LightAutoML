import torch
import torch.nn as nn


class TS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) + torch.sigmoid(x - 10) + torch.sigmoid(x - 20)
