import fire
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy

from calibration import calibrator_dict
from utils import (LABELS_DIR,
                   LOGITS_DIR,
                   WEIGHTS_DIR,
                   RESULTS_DIR,
                   DEFAULT_OUTPUT_PATH,
                   DEFAULT_DATASET_PATH,
                   DEVICE,
                   get_logits_and_labels)
from pathlib import Path
from tabulate import tabulate


def main(calibrator_name: str,
         dataset_name: str="ImageNet_Val",
         model_type_name: str="HuggingfaceModel",
         weights_name: str="openai/clip-vit-large-patch14",
         dataset_path: str=DEFAULT_DATASET_PATH,
         output_path: str=DEFAULT_OUTPUT_PATH,
         **kwargs):
    print(f"device: {DEVICE}")
    output_path = Path(output_path)

    weights_name = weights_name.replace("/","@")

    logits = torch.load(str(output_path / LOGITS_DIR / dataset_name / model_type_name / f"{weights_name}.pt"))
    labels = torch.load(str(output_path / LABELS_DIR / f"{dataset_name}.pt")).long()

    kwargs["length_logits"] = logits.shape[1]
    calibrator = calibrator_dict[calibrator_name](**kwargs)
    calibrator.tune(logits, labels)

    # Save weights to local files.
    (output_path / WEIGHTS_DIR / dataset_name / model_type_name).mkdir(parents=True, exist_ok=True)
    calibrator.save_model(str(output_path / WEIGHTS_DIR / dataset_name / calibrator_name / model_type_name / f"{weights_name.replace('/', '@')}.pt"))

    calibration_model = calibrator.get_model().to(DEVICE)
    logits_dataset = TensorDataset(logits, labels)
    logits_dataloader = DataLoader(logits_dataset, batch_size=128)
    calibrated_logits, _ = get_logits_and_labels(calibration_model,
                                                 logits_dataloader,
                                                 device=DEVICE)  # Logits after calibration.

    calibrated_logits = calibrated_logits.to(DEVICE)
    labels = labels.to(DEVICE)
    logits = logits.to(DEVICE)
    ece_criterion = MulticlassCalibrationError(1000).to(DEVICE)
    accuracy_criterion = MulticlassAccuracy(1000).to(DEVICE)

    # Compute loss and other metrics before and after calibration.
    results = {}
    loss_fn = calibrator.get_loss_func().to(DEVICE)
    results["loss_fn_name"] = loss_fn.__class__.__name__
    results["loss_before"] = loss_fn(logits, labels).item()
    results["ece_before"] = ece_criterion(logits, labels).item()
    results["acc_before"] = accuracy_criterion(logits, labels).item()
    results["loss_after"] = loss_fn(calibrated_logits, labels).item()
    results["ece_after"] = ece_criterion(calibrated_logits, labels).item()
    results["acc_after"] = accuracy_criterion(calibrated_logits, labels).item()

    print("Saving results.")
    calibration_dir = output_path / RESULTS_DIR / "calib_dset" / model_type_name
    calibration_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(calibration_dir / f"{weights_name}_{calibrator_name}.pt"))

    headers = ["", results["loss_fn_name"], "ECE", "Accuracy"]
    table = [["Before Calibration", results["loss_before"], results["ece_before"], results["acc_before"]],
             ["After Calibration", results["loss_after"], results["ece_after"], results["acc_after"]]]
    print(tabulate(table, headers, tablefmt="simple_outline"))


if __name__ == "__main__":
    fire.Fire(main)