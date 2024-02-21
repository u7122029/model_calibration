from .huggingface_model import HuggingfaceModel
from .base_backbone import Backbone
from .lavis_model import BlipModel, ALBEFModel
import sys, inspect


def get_class_bases(x):
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases

base_model_dict = inspect.getmembers(sys.modules[__name__],
                                     lambda x: inspect.isclass(x) and Backbone in get_class_bases(x))
base_model_dict = {x: y for x, y in base_model_dict}