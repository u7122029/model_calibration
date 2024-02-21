import torch
from torch import nn
from abc import ABC, abstractmethod
from pathlib import Path


def save_model_pytorch(model: nn.Module, filepath: str):
    """
        Saves the pytorch _model weights.
        :param model: The pytorch _model.
        :param filepath: The path to the file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)

    print(f"Saving _model to: {filepath}")
    torch.save(model.state_dict(), str(filepath))


def load_model_pytorch(model: nn.Module, filepath: str):
    """
    Loads the pytorch _model weights.
    :param model: The _model to load the weights into
    :param filepath: The path to the weights file.
    """
    filepath = Path(filepath)
    print(f"Loading _model from: {filepath}")

    checkpoint = torch.load(str(filepath))
    model.load_state_dict(checkpoint)


class Calibrator(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def tune(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def load_model(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()