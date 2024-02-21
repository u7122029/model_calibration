from transformers import AutoProcessor, AutoModel
import torch.nn as nn
from transformers import CLIPModel
from .base_backbone import Backbone


class HuggingfaceModel(Backbone):
    def __init__(self, weights_name: str, label_texts: list[str], device="cpu"):
        super().__init__()
        self.weights_name = weights_name
        self.processor = AutoProcessor.from_pretrained(self.weights_name)
        self.label_texts = label_texts
        self.model = AutoModel.from_pretrained(self.weights_name)
        self.device = device
        self.to(self.device)

    def forward(self, x: list):
        x = self.processor(images=x, text=self.label_texts, return_tensors="pt", padding=True).to(self.device)
        x = self.model(**x).logits_per_image
        return x