from abc import ABC

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import relu, one_hot
from tqdm import tqdm
from torchmetrics import MeanMetric

from .base_calibrator import save_model_pytorch, load_model_pytorch, Calibrator
from .criterions import OneHotMSE


class PTSModel(nn.Module):
    def __init__(self, length_logits: int, hidden_layers: int=2,
                 nodes_per_layer: int=5, top_k_logits: int=10, **kwargs):
        """
        Initialiser for a Parameterised Temperature Scaling (PTS) Model.
        :param length_logits: The length of the input logits.
        :param hidden_layers: The number of hidden layers.
        :param nodes_per_layer: The number of nodes in each hidden layer.
        :param top_k_logits: The top k logits to take from each logit vector.
        :param kwargs: Extra args to not be used.
        """
        super().__init__()
        assert hidden_layers >= 0

        self.hidden_layers = hidden_layers
        self.nodes_per_layer = nodes_per_layer
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        #Build _model
        self.layers = []
        if self.hidden_layers == 0:
            self.layers.append(nn.Linear(in_features=self.top_k_logits, out_features=1))
        else:
            self.layers.append(nn.Linear(in_features=self.top_k_logits, out_features=self.nodes_per_layer))

        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(in_features=self.nodes_per_layer, out_features=self.nodes_per_layer))

        if self.hidden_layers > 0:
            self.layers.append(nn.Linear(in_features=self.nodes_per_layer, out_features=1))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, inp: torch.Tensor):
        t = torch.topk(inp, self.top_k_logits, dim=1).values
        t = self.layers[0](t)
        if len(self.layers) > 0:
            t = relu(t)

        for layer_idx in range(1, len(self.layers) - 1):
            t = self.layers[layer_idx](t)
            t = relu(t)

        if len(self.layers) > 0:
            t = self.layers[-1](t)

        t = torch.clip(torch.abs(t),1e-12,1e12)

        x = inp / t
        x = torch.softmax(x, dim=1)
        return x


class PTSCalibrator(Calibrator):
    """
    Class for Parameterized Temperature Scaling (PTS) using PyTorch.
    """

    def __init__(self,
                 length_logits,
                 epochs=1000,
                 lr=5e-5,
                 weight_decay=0,
                 batch_size=1000,
                 hidden_layers=2,
                 nodes_per_layer=5,
                 top_k_logits=10):
        """
        Initialiser for the Parameterised Temperature Scaling (PTS) model.
        :param length_logits: Lengths of logits vector.
        :param epochs: Number of epochs for tuning.
        :param lr: Learning rate
        :param weight_decay: Weight decay
        :param batch_size: Batch size
        :param hidden_layers: Number of hidden layers in the network.
        :param nodes_per_layer: Number of nodes in each hidden layer.
        :param top_k_logits: Top k logits to take from each logit vector.
        """
        super().__init__()
        assert hidden_layers >= 0

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = hidden_layers
        self.nodes_per_layer = nodes_per_layer
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        self._model = PTSModel(self.length_logits, self.nlayers, self.nodes_per_layer, self.top_k_logits)

    def tune(self, logits, labels, device="cpu"):
        """
        Tunes the PTS model.
        :param logits: The full set of logits.
        :param labels: The full set of labels.
        :param device: The device that the model should be put on.
        """
        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"

        dset = TensorDataset(logits, labels)
        criterion = self.get_loss_func().to(device)
        optimiser = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._model.train()
        self._model = self._model.to(device)
        dataloader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        mean_loss = MeanMetric()

        epoch_progress = tqdm(range(self.epochs), desc="PTSCalibrator Epoch")
        for _ in epoch_progress:
            epoch_loss = 0
            for logits_batch, labels_batch in dataloader:
                optimiser.zero_grad()

                logits_batch = logits_batch.to(device)
                labels_batch = labels_batch.to(device)
                outs = self._model(logits_batch)
                loss = criterion(outs, labels_batch)
                epoch_loss += loss.item()

                loss.backward()
                optimiser.step()

            mean_loss.update(epoch_loss)
            epoch_progress.set_postfix({"mean_loss": mean_loss.compute().item()})

    def save_model(self, filepath: str): save_model_pytorch(self._model, filepath)

    def load_model(self, filepath: str): load_model_pytorch(self._model, filepath)

    def get_loss_func(self):
        return OneHotMSE()

    def get_model(self): return self._model