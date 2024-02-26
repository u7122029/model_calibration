import inspect
import sys

from torchvision.datasets import VisionDataset

from .imagenet_variants import (ImageNet_Val,
                                get_imagenet_dset,
                                get_imagenet_o_mask,
                                get_imagenet_a_mask,
                                ImageNetO,
                                ImageNetA)


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