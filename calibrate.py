import fire
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy

from calibration import calibrator_dict
from utils import (DEFAULT_LABELS_PATH,
                   DEFAULT_LOGITS_PATH,
                   DEFAULT_WEIGHTS_PATH,
                   DEFAULT_DATASET_PATH,
                   DEVICE,
                   get_logits_and_labels)
from pathlib import Path


def main(calibrator_name: str,
         dataset_name: str="ImageNet",
         model_type_name: str= "HuggingfaceModel",
         weights_name: str="openai/clip-vit-large-patch14",
         weights_path: str=DEFAULT_WEIGHTS_PATH,
         logits_path: str=DEFAULT_LOGITS_PATH,
         labels_path: str=DEFAULT_LABELS_PATH,
         dataset_path: str=DEFAULT_DATASET_PATH,
         **kwargs):
    print(f"device: {DEVICE}")
    weights_path = Path(weights_path)
    logits_path = Path(logits_path)
    labels_path = Path(labels_path)
    dataset_path = Path(dataset_path)

    weights_name = weights_name.replace("/","@")

    calibrator = calibrator_dict[calibrator_name](**kwargs)
    logits = torch.load(str(logits_path / dataset_name / model_type_name / f"{weights_name}.pt"))
    print(logits.shape)
    labels = torch.load(str(labels_path / f"{dataset_name}.pt")).long()
    calibrator.tune(logits, labels)

    # Save weights to local files.
    (weights_path / dataset_name / model_type_name).mkdir(parents=True, exist_ok=True)
    calibrator.save_model(str(weights_path / dataset_name / calibrator_name / model_type_name / f"{weights_name.replace('/', '@')}.pt"))

    calibration_model = calibrator.get_model().to(DEVICE)
    logits_dataset = TensorDataset(logits, labels)
    logits_dataloader = DataLoader(logits_dataset, batch_size=128)
    calibrated_logits, _ = get_logits_and_labels(calibration_model,
                                                 logits_dataloader,
                                                 device=DEVICE)  # Logits after calibration.

    calibrated_logits = calibrated_logits.to(DEVICE)
    labels = labels.to(DEVICE)
    logits = logits.to(DEVICE)
    nll_criterion = nn.CrossEntropyLoss().to(DEVICE)
    ece_criterion = MulticlassCalibrationError(1000).to(DEVICE)
    accuracy_criterion = MulticlassAccuracy(1000).to(DEVICE)

    print(nll_criterion(logits, labels).item(), ece_criterion(logits, labels).item(),
          accuracy_criterion(logits, labels).item())
    print(nll_criterion(calibrated_logits, labels).item(), ece_criterion(calibrated_logits, labels).item(),
          accuracy_criterion(calibrated_logits, labels).item())


if __name__ == "__main__":
    fire.Fire(main)