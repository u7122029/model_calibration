import torch
from backbone_models import Backbone
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, Imagenette, VisionDataset
from torchvision.datasets import vision
import sys, inspect

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_WEIGHTS_PATH = "weights"
DEFAULT_LOGITS_PATH = "logits"
DEFAULT_LABELS_PATH = "labels"
DEFAULT_DATASET_PATH = "C:/ml_datasets"
DEFAULT_RESULTS_PATH = "results"


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


def get_class_bases(x):
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases


def dset_class_predicate(x):
    if not inspect.isclass(x): return False

    class_bases = get_class_bases(x)
    return VisionDataset in class_bases #or vision.VisionDataset in class_bases


dataset_dict = inspect.getmembers(sys.modules[__name__], dset_class_predicate)
dataset_dict = {x: y for x, y in dataset_dict}