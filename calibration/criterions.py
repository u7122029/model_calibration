from torch import nn
import torch
import torch.nn.functional as F


class OneHotMSE(nn.MSELoss):
    """
    Simply an nn.MSELoss criterion that applies a one-hot encoding to the labels before computing the loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inp: torch.Tensor, labels: torch.Tensor):
        labels = F.one_hot(labels,inp.shape[1]).float()
        return super().forward(inp, labels)