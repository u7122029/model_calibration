import sys, inspect
from .temperature_scaling import TemperatureScalingCalibrator
from .parametric_temperature_scaling import PTSCalibrator
from .base_calibrator import Calibrator

from torch import nn


class ModelWithCalibration(nn.Module):
    def __init__(self, backbone_model: nn.Module, calibration_model: nn.Module):
        super().__init__()
        self.backbone_model = backbone_model
        self.calibration_model = calibration_model

    def forward(self, x):
        return self.calibration_model(self.backbone_model(x))


# Create a dictionary that maps the name of each calibrator class as a string to the class itself.
calibrator_dict = inspect.getmembers(sys.modules[__name__], lambda x: inspect.isclass(x) and Calibrator in x.__bases__)
calibrator_dict = {x: y for x, y in calibrator_dict}
