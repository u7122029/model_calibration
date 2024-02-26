from torchvision.datasets import ImageNet, ImageFolder
import fire
from utils import DEFAULT_MASKS_PATH, DEFAULT_LABELS_PATH
from pathlib import Path


def imagenet_val_masks():
    labels_path = Path(DEFAULT_LABELS_PATH) / "ImageNet_val.pt"
    if not labels_path.exists():
        print("Failed to generate ImageNet validation set masks - ImageNet_val.pt does not exist.")
        return
    # The labels do exist.



def main():
    # We need to generate all the masks.
    pass


if __name__ == "__main__":
    fire.Fire(main)