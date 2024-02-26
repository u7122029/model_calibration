import json
from pathlib import Path

import fire
import torch
from torch.utils.data import DataLoader

from backbone_models import base_model_dict
from data import dataset_dict
from utils import (DEVICE,
                   get_logits_and_labels,
                   LOGITS_DIR,
                   WEIGHTS_DIR,
                   LABELS_DIR,
                   DEFAULT_DATASET_PATH,
                   DEFAULT_OUTPUT_PATH,
                   list_collate_fn)


def a_or_an(word):
    vowels = "aeiouy"
    if word[0] in vowels:
        return "an"
    return "a"


def main(weights_name: str = "openai/clip-vit-large-patch14",
         model_type_name: str = "HuggingfaceModel",
         dataset_name: str = "ImageNet_Val",
         output_dir: str = DEFAULT_OUTPUT_PATH,
         datasets_dir: str = DEFAULT_DATASET_PATH):
    print(f"device: {DEVICE}")
    output_dir = Path(output_dir)
    datasets_dir = Path(datasets_dir)

    (output_dir / WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    (output_dir / LOGITS_DIR).mkdir(parents=True, exist_ok=True)
    (output_dir / LABELS_DIR).mkdir(parents=True, exist_ok=True)
    # dataset_path.mkdir(parents=True, exist_ok=True)

    dataset_class = dataset_dict[dataset_name]
    dataset = dataset_class(str(datasets_dir / "ImageNet"), split="calib")
    label_texts = [f"a picture of {a_or_an(label)} {label}" for label in json.load(open("imagenet-simple-labels.json"))]

    print(f"sample label: {label_texts[0]}")

    base_model = base_model_dict[model_type_name](weights_name, label_texts, DEVICE)

    dataloader = DataLoader(dataset, batch_size=64, collate_fn=list_collate_fn)
    logits, labels = get_logits_and_labels(base_model, dataloader,
                                           DEVICE)  # calibrator.tune(dataloader, DEVICE) # Logits before calibration.
    labels = labels.long()

    # Save labels to pytorch file.
    torch.save(labels.cpu(), str(output_dir / LABELS_DIR / f"{dataset_name}.pt"))

    # Save logits to pytorch file.
    (output_dir / LOGITS_DIR / dataset_name / model_type_name).mkdir(parents=True, exist_ok=True)
    torch.save(logits.cpu(),
               str(output_dir / LOGITS_DIR / dataset_name / model_type_name / f"{weights_name.replace('/', '@')}.pt"))


if __name__ == "__main__":
    fire.Fire(main)
