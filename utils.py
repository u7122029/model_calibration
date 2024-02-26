import torch
from backbone_models import Backbone
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, Imagenette, VisionDataset, ImageFolder
from torchvision.datasets import vision
import sys, inspect

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUTPUT_PATH = "out"
WEIGHTS_DIR = "weights"
LOGITS_DIR = "logits"
LABELS_DIR = "labels"
RESULTS_DIR = "results"
DEFAULT_MASKS_PATH = f"{DEFAULT_OUTPUT_PATH}/masks"
DEFAULT_DATASET_PATH = "C:/ml_datasets"


def get_logits_and_labels(model: nn.Module, dataloader: DataLoader, device=DEVICE):
    logits_lst = []
    labels_lst = []

    model.eval() # Ensure that the model is on eval mode.
    with torch.no_grad():
        for batch, labels in tqdm(dataloader):
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)

            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            logits = model(batch)
            logits_lst.append(logits.cpu())
            labels_lst.append(labels.cpu())

    logits_out = torch.cat(logits_lst).cpu()
    labels_out = torch.cat(labels_lst).cpu()
    return logits_out, labels_out


def list_collate_fn(items):
    batch = []
    labels = []
    for inp, label in items:
        batch.append(inp)
        labels.append(label)
    return batch, torch.Tensor(labels)