from torchvision.datasets import ImageNet
import fire

from torchmetrics.classification import MulticlassCalibrationError, Accuracy, MulticlassAccuracy
import json
from backbone_models import base_model_dict
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
import torch.nn as nn
from pathlib import Path
from calibration import calibrator_dict, ModelWithCalibration
from utils import (DEVICE,
                   get_logits_and_labels,
                   DEFAULT_LOGITS_PATH,
                   DEFAULT_WEIGHTS_PATH,
                   DEFAULT_LABELS_PATH,
                   DEFAULT_DATASET_PATH,
                   dataset_dict)


def a_or_an(word):
    vowels = "aeiouy"
    if word[0] in vowels:
        return "an"
    return "a"


def id_collate(x):
    batch = [a[0] for a in x]
    labels = torch.Tensor([a[1] for a in x])
    return batch, labels


def main(weights_name: str="openai/clip-vit-large-patch14",
         model_type_name: str="HuggingfaceModel",
         dataset_name: str="ImageNet",
         weights_path: str=DEFAULT_WEIGHTS_PATH,
         logits_path: str=DEFAULT_LOGITS_PATH,
         labels_path: str=DEFAULT_LABELS_PATH,
         dataset_path: str=DEFAULT_DATASET_PATH):
    print(f"device: {DEVICE}")
    weights_path = Path(weights_path)
    logits_path = Path(logits_path)
    labels_path = Path(labels_path)
    dataset_path = Path(dataset_path)

    weights_path.mkdir(parents=True, exist_ok=True)
    logits_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)

    dataset_class = dataset_dict[dataset_name]
    dataset = dataset_class(str(dataset_path / dataset_name), split="val")
    label_texts = [f"a picture of {a_or_an(label)} {label}" for label in json.load(open("imagenet-simple-labels.json"))]
    # TODO: HANDLE THE DIFFERENT CLASSES FOR EACH DATASET.

    print(f"sample label: {label_texts[0]}")

    base_model = base_model_dict[model_type_name](weights_name, label_texts, DEVICE)

    dataloader = DataLoader(dataset, batch_size=64, collate_fn=id_collate)
    logits, labels = get_logits_and_labels(base_model, dataloader, DEVICE)#calibrator.tune(dataloader, DEVICE) # Logits before calibration.
    labels = labels.long()

    # Save labels to pytorch file.
    torch.save(labels.cpu(), str(labels_path / f"{dataset_name}.pt"))

    # Save logits to pytorch file.
    (logits_path / dataset_name / model_type_name).mkdir(parents=True, exist_ok=True)
    torch.save(logits.cpu(), str(logits_path / dataset_name / model_type_name / f"{weights_name.replace('/', '@')}.pt"))


if __name__ == "__main__":
    fire.Fire(main)