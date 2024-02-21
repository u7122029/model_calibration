from abc import ABC, abstractmethod
from torch import nn


class Backbone(ABC, nn.Module):
    """
    Abstract class that groups together all types of backbones.
    """
    @abstractmethod
    def __init__(self):
        super().__init__()
