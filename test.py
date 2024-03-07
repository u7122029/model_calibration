import json

import fire
import torch
from data import (get_imagenet_o_mask,
                  get_imagenet_a_mask,
                  ImageNet_Val,
                  ImageNetO,
                  ImageNetA,
                  ImageNetC_Blur,
                  ImageNetC_Digital,
                  ImageNetC_Extra,
                  ImageNetC_Noise,
                  ImageNetC_Weather)
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


def get_logits(model: ModelWithCalibration, dataloader: DataLoader, mask: torch.Tensor=None, desc: str=None):
    model.eval()
    backbone_model = model.backbone_model
    calibration_model = model.calibration_model

    uncalibrated_outs = []
    calibrated_outs = []
    all_labels = []

    with torch.no_grad():
        for batch, labels in tqdm(dataloader, desc=desc):
            all_labels.append(labels)
            uncalibrated_logits = backbone_model(batch)
            calibrated_logits = calibration_model(uncalibrated_logits)

            if mask is not None:
                uncalibrated_logits = uncalibrated_logits[:, mask]
                calibrated_logits = calibrated_logits[:, mask]

            uncalibrated_outs.append(uncalibrated_logits.cpu())
            calibrated_outs.append(calibrated_logits.cpu())

    return torch.cat(uncalibrated_outs), torch.cat(calibrated_outs), torch.cat(all_labels)


def eval_id_ood(model: ModelWithCalibration): # Can either be a model with backbone or just a model.
    id_dataset = ImageNet_Val(str(Path(DEFAULT_DATASET_PATH) / "ImageNet"), split="test")
    ood_dataset = ImageNetO(str(Path(DEFAULT_DATASET_PATH) / "ImageNetO"))
    id_dataloader = DataLoader(id_dataset, batch_size=64, collate_fn=list_collate_fn)
    ood_dataloader = DataLoader(ood_dataset, batch_size=64, collate_fn=list_collate_fn)
    mask = get_imagenet_o_mask()

    id_confidences_uncalibrated = []
    ood_confidences_uncalibrated = []
    id_confidences_calibrated = []
    ood_confidences_calibrated = []

    id_logits_uncalibrated, id_logits_calibrated, _ = get_logits(model,
                                                                 id_dataloader,
                                                                 mask,
                                                                 desc="ImageNet id Eval")
    ood_logits_uncalibrated, ood_logits_calibrated, _ = get_logits(model,
                                                                   ood_dataloader,
                                                                   mask,
                                                                   desc="ImageNet ood Eval")

    id_probs_uncalibrated = torch.softmax(id_logits_uncalibrated, 1).cpu()
    id_probs_calibrated = torch.softmax(id_logits_calibrated, 1).cpu()

    id_confidences_uncalibrated.append(id_probs_uncalibrated.max(1).values)
    id_confidences_calibrated.append(id_probs_calibrated.max(1).values)

    ood_probs_uncalibrated = torch.softmax(ood_logits_uncalibrated, 1).cpu()
    ood_probs_calibrated = torch.softmax(ood_logits_calibrated, 1).cpu()

    ood_confidences_uncalibrated.append(ood_probs_uncalibrated.max(1).values)
    ood_confidences_calibrated.append(ood_probs_calibrated.max(1).values)

    id_scores_uncalibrated = -torch.cat(id_confidences_uncalibrated)
    ood_scores_uncalibrated = -torch.cat(ood_confidences_uncalibrated)
    id_scores_calibrated = -torch.cat(id_confidences_calibrated)
    ood_scores_calibrated = -torch.cat(ood_confidences_calibrated)

    scores_uncalibrated = torch.cat([id_scores_uncalibrated, ood_scores_uncalibrated])
    scores_calibrated = torch.cat([id_scores_calibrated, ood_scores_calibrated])

    labels = torch.zeros(len(scores_uncalibrated))
    labels[:len(id_dataset)] = 1
    labels = labels.int()

    auroc = AUROC(task="binary")
    auroc_score_before = auroc(scores_uncalibrated, labels)
    auroc_score_after = auroc(scores_calibrated, labels)

    avg_precision = AveragePrecision(task="binary")
    avg_precision_before = avg_precision(scores_uncalibrated, labels)
    avg_precision_after = avg_precision(scores_calibrated, labels)
    return auroc_score_before, auroc_score_after, avg_precision_before, avg_precision_after


def eval_imagenetA(model: ModelWithCalibration):
    mask = get_imagenet_a_mask()
    dset = ImageNetA(str(Path(DEFAULT_DATASET_PATH) / "ImageNetA"))
    dataloader = DataLoader(dset, batch_size=64, collate_fn=list_collate_fn)
    accuracy = Accuracy("multiclass", num_classes=200).to(DEVICE)
    ece = CalibrationError("multiclass", num_classes=200).to(DEVICE)

    uncalibrated_logits, calibrated_logits, all_labels = get_logits(model, dataloader, mask, "ImageNetA Eval")
    uncalibrated_preds = torch.max(uncalibrated_logits, dim=1).indices
    calibrated_preds = torch.max(calibrated_logits, dim=1).indices

    uncalibrated_probs = torch.softmax(uncalibrated_logits, dim=1)
    calibrated_probs = torch.softmax(calibrated_logits, dim=1)

    uncalibrated_accuracy = accuracy(uncalibrated_preds, all_labels)
    calibrated_accuracy = accuracy(calibrated_preds, all_labels)

    uncalibrated_ece = ece(uncalibrated_probs, all_labels)
    calibrated_ece = ece(calibrated_probs, all_labels)

    return uncalibrated_accuracy, calibrated_accuracy, uncalibrated_ece, calibrated_ece


def eval_imagenetC(model: ModelWithCalibration):
    path = Path(DEFAULT_DATASET_PATH) / "ImageNetC"
    dset_blur = ImageNetC_Blur(str(path))
    dset_digital = ImageNetC_Digital(str(path))
    dset_extra = ImageNetC_Extra(str(path))
    dset_noise = ImageNetC_Noise(str(path))
    dset_weather = ImageNetC_Weather(str(path))

    dsets = [dset_blur, dset_digital, dset_extra, dset_noise, dset_weather]
    accuracy_metric = Accuracy("multiclass", num_classes=1000)
    ece_metric = CalibrationError("multiclass", num_classes=1000)
    uncalibrated_accs = []
    calibrated_accs = []
    uncalibrated_eces = []
    calibrated_eces = []
    for dset in dsets:
        dataloader = DataLoader(dset, batch_size=64, collate_fn=list_collate_fn)
        uncalibrated_logits, calibrated_logits, labels = get_logits(model,
                                                                    dataloader,
                                                                    desc=f"{dset.__class__.__name__} Eval")
        uncalibrated_preds = torch.max(uncalibrated_logits, dim=1).indices
        calibrated_preds = torch.max(calibrated_logits, dim=1).indices

        uncalibrated_probs = torch.softmax(uncalibrated_logits, dim=1)
        calibrated_probs = torch.softmax(calibrated_logits, dim=1)

        uncalibrated_acc = accuracy_metric(uncalibrated_preds, labels)
        calibrated_acc = accuracy_metric(calibrated_preds, labels)

        uncalibrated_ece = ece_metric(uncalibrated_probs, labels)
        calibrated_ece = ece_metric(calibrated_probs, labels)

        uncalibrated_accs.append(uncalibrated_acc)
        calibrated_accs.append(calibrated_acc)
        uncalibrated_eces.append(uncalibrated_ece)
        calibrated_eces.append(calibrated_ece)

    return uncalibrated_accs, calibrated_accs, uncalibrated_eces, calibrated_eces


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
    uncalibrated_accs, calibrated_accs, uncalibrated_eces, calibrated_eces = eval_imagenetC(calibrated_model)
    """acc_before, acc_after, ece_before, ece_after = eval_imagenetA(calibrated_model)
    auroc_before, auroc_after, avg_precision_before, avg_precision_after = eval_id_ood(calibrated_model)
    

    headers = ["ImageNet-O/val ID/OOD Classification", "AUROC", "Avg. Precision"]
    table = [["Uncalibrated", auroc_before, avg_precision_before],
             ["Calibrated", auroc_after, avg_precision_after]]
    print(tabulate(table, headers=headers, tablefmt="simple_outline"))

    headers = ["ImageNet-A Classification", "Accuracy", "ECE"]
    table = [["Uncalibrated", acc_before, ece_before],
             ["Calibrated", acc_after, ece_after]]
    print(tabulate(table, headers=headers, tablefmt="simple_outline"))"""

    headers = ["ImageNet-C Classification", "ImageNet_Blur", "ImageNet_Digital", "ImageNet_Blur", "ImageNet_Blur", "ImageNet_Blur"]
    table = [["Uncalibrated ECE"] + uncalibrated_eces,
             ["Calibrated ECE"] + calibrated_eces,
             ["Uncalibrated Acc"] + uncalibrated_accs,
             ["Calibrated Acc"] + calibrated_accs]
    print(tabulate(table, headers=headers, tablefmt="simple_outline"))

if __name__ == "__main__":
    fire.Fire(main)