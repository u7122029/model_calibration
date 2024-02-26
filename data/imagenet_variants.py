from torchvision.datasets import ImageFolder, ImageNet, VisionDataset
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path
import json
import torch


def get_imagenet_o_mask():
    with open("imagenet-o-mask.json") as f:
        lst = json.load(f)
    return lst


def get_imagenet_a_mask():
    with open("imagenet-a-mask.json") as f:
        lst = json.load(f)
    return lst



class ImageNet_Val(Subset, VisionDataset):
    def __init__(self, root: str, split: str, **kwargs):
        """

        :param root: ImageNet dataset root path.
        :param split: Either "calib" for calibration, or "test" for testing.
        :param label_type: Either "class" for regular image classification, or
                            "dist" for in/out distribution classification.
        :param kwargs:
        """

        dset = ImageNet(root, split="val")

        imagenet_o_class_mask = torch.Tensor(get_imagenet_o_mask()).int()
        imagenet_o_mask = imagenet_o_class_mask.bool().repeat_interleave(50)

        all_class_indices = (torch.arange(50000) % 50) < 25

        mask = ~(imagenet_o_mask & all_class_indices)

        if split == "test":
            mask = ~mask
        elif split not in {"calib", "test"}:
            raise ValueError(f"split \"{split}\" is not in ['calib', 'test'].")

        super().__init__(dset, torch.arange(50000)[mask])


class ImageNetO(ImageFolder):
    def __init__(self, root: str, label_type: str="class", **kwargs):
        target_transform = None
        if label_type == "dist":
            target_transform = lambda x: 0  # 0 for out-of-distribution.
        elif label_type not in {"dist", "class"}:
            raise ValueError(f"label_type \"{label_type}\" not in ['dist', 'class'].")
        super().__init__(root, target_transform=target_transform, **kwargs)


class ImageNetO_Dist(ConcatDataset):
    """
    Imagenet-O in/out of distribution classification dataset.
    """
    def __init__(self, imagenet_root: str, imagenet_o_root: str):
        super().__init__([
            ImageNet_Val(imagenet_root, "test"),
            ImageNetO(imagenet_o_root, "dist")
        ])


class ImageNetA(ImageFolder):
    def __init__(self, imagenet_a_root: str, **kwargs):
        super().__init__(imagenet_a_root, **kwargs)


def get_imagenet_dset(imagenet_root: str, imagenet_o_root: str=None, version: str="calib"):
    """

    :param imagenet_root: root path of ImageNet dataset.
    :param imagenet_o_root:
    :param version: "calib" for calibration, or "dist" for testing.
    :return:
    """
    if version == "calib":
        return ImageNet_Val(imagenet_root,"calib")
    elif version == "test_dist":
        assert imagenet_o_root is not None, "imagenet_o_root should not be None."
        return ImageNetO_Dist(imagenet_root, imagenet_o_root)
    elif version == "test_class":
        return ImageNet_Val(imagenet_root, "test")
    elif version == "imagenet_a":
        return ImageNetA(imagenet_root)

    assert False, "invalid inputs."
