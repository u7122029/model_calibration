from .base_calibrator import Calibrator
from torch import nn


class IdentityCalibrator(Calibrator):
    """
    Calibrator that does not do anything to the logits.
    Should not require tuning.
    """
    def __init__(self, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def get_loss_func(self):
        pass

    def get_model(self):
        return nn.Identity()

    def tune(self, *args, **kwargs):
        pass