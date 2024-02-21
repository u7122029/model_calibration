import torch
from torch import nn, optim

from .base_calibrator import Calibrator, save_model_pytorch, load_model_pytorch


class TemperatureScalingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


class TemperatureScalingCalibrator(Calibrator):
    def __init__(self, lr=0.015, lbfgs_max_iter=200, **kwargs):
        """
        Initialiser for a Temperature Scaling Calibrator.
        :param lr: The learning rate.
        :param lbfgs_max_iter: The maximum number of iterations to use in the L-BFGS optimiser.
        :param kwargs: Extra args to not be used.
        """
        super().__init__()
        self.lr = lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.model = TemperatureScalingModel()

    def tune(self, logits, labels, device="cpu"):
        nll_criterion = nn.CrossEntropyLoss().to(device)
        self.model = self.model.to(device)
        optimizer = optim.LBFGS(self.model.parameters(), lr=self.lr, max_iter=self.lbfgs_max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.model(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

    def save_model(self, filepath: str): save_model_pytorch(self.model, filepath)

    def load_model(self, filepath: str): load_model_pytorch(self.model, filepath)

    def get_model(self): return self.model
