import json

import fire
import torch
from data import get_imagenet_o_mask, get_imagenet_a_mask, ImageNet_Val, ImageNetO, ImageNetA
from calibration import ModelWithCalibration, calibrator_dict
from utils import DEFAULT_DATASET_PATH, DEVICE, WEIGHTS_DIR, list_collate_fn, DEFAULT_OUTPUT_PATH
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torchmetrics import AUROC, AveragePrecision, Accuracy, CalibrationError
from typing import Union
from backbone_models import base_model_dict
from tabulate import tabulate


def eval_id_ood(model: nn.Module): # Can either be a model with backbone or just a model.
    id_dataset = ImageNet_Val(str(Path(DEFAULT_DATASET_PATH) / "ImageNet"), split="test")
    ood_dataset = ImageNetO(str(Path(DEFAULT_DATASET_PATH) / "ImageNetO"))
    id_dataloader = DataLoader(id_dataset, batch_size=64, collate_fn=list_collate_fn)
    ood_dataloader = DataLoader(ood_dataset, batch_size=64, collate_fn=list_collate_fn)
    mask = get_imagenet_o_mask()

    id_confidences = []
    ood_confidences = []

    model.eval()
    with torch.no_grad():
        for id_batch, _ in tqdm(id_dataloader, desc="imagenet id"):
            outs = model(id_batch)[:,mask] # Imagenet-O Logits.
            probs = torch.softmax(outs, 1).cpu()
            id_confidences.append(probs.max(1).values)

        for ood_batch, _ in tqdm(ood_dataloader, desc="imagenet ood"):
            outs = model(ood_batch)[:, mask]  # Imagenet-O Logits.
            probs = torch.softmax(outs,1).cpu()
            ood_confidences.append(probs.max(1).values)

    id_scores = -torch.cat(id_confidences)
    ood_scores = -torch.cat(ood_confidences)
    scores = torch.cat([id_scores, ood_scores])

    labels = torch.zeros(len(scores))
    labels[:len(id_scores)] = 1
    labels = labels.int()

    auroc = AUROC(task="binary")
    auroc_score = auroc(scores, labels)

    avg_precision = AveragePrecision(task="binary")
    avg_precision_score = avg_precision(scores, labels)
    return auroc_score, avg_precision_score


def eval_imagenetA(model: nn.Module):
    mask = get_imagenet_a_mask()
    dset = ImageNetA(str(Path(DEFAULT_DATASET_PATH) / "ImageNetA"))
    dataloader = DataLoader(dset, batch_size=64, collate_fn=list_collate_fn)
    accuracy = Accuracy("multiclass", num_classes=200).to(DEVICE)
    ece = CalibrationError("multiclass", num_classes=200).to(DEVICE)

    with torch.no_grad():
        for batch, labels in tqdm(dataloader):
            labels = labels.to(DEVICE)
            outs = model(batch)[:, mask].to(DEVICE)  # Imagenet-O Logits.
            preds = torch.max(outs, dim=1).indices
            accuracy.update(preds, labels)
            ece.update(outs, labels)

    return accuracy.compute(), ece.compute()


def main(backbone_model_type: str="HuggingfaceModel",
         weights_name: str="openai/clip-vit-large-patch14",
         calibrator_name: Union[None, str]="PTSCalibrator",
         dataset_name: str="ImageNet_Val",
         length_logits: int=1000):

    with open("imagenet-simple-labels.json") as f:
        label_texts = json.load(f)

    backbone_model = base_model_dict[backbone_model_type](weights_name, label_texts, DEVICE)
    calibrator = calibrator_dict[calibrator_name](length_logits=length_logits)
    calibrator.load_model(str(Path(DEFAULT_OUTPUT_PATH) /
                              WEIGHTS_DIR /
                              dataset_name /
                              calibrator_name /
                              backbone_model_type /
                              f"{weights_name.replace('/', '@')}.pt"))
    calibrated_model = ModelWithCalibration(backbone_model,
                                 calibrator.get_model())
    calibrated_model = calibrated_model.to(DEVICE)

    # Test the given calibrator over all the datasets.
    acc_before, ece_before = eval_imagenetA(backbone_model)
    acc_after, ece_after = eval_imagenetA(calibrated_model)

    auroc_before, avg_precision_before = eval_id_ood(backbone_model)
    auroc_after, avg_precision_after = eval_id_ood(calibrated_model)

    headers = ["ImageNet-O/val ID/OOD Classification", "AUROC", "Avg. Precision"]
    table = [["Before Calibration", auroc_before, avg_precision_before],
             ["After Calibration", auroc_after, avg_precision_after]]
    print(tabulate(table, headers=headers, tablefmt="simple_outline"))

    headers = ["ImageNet-A Classification", "Accuracy", "ECE"]
    table = [["Before Calibration", acc_before, ece_before],
             ["After Calibration", acc_after, ece_after]]
    print(tabulate(table, headers=headers, tablefmt="simple_outline"))


if __name__ == "__main__":
    fire.Fire(main)