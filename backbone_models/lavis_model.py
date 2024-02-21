from .base_backbone import Backbone
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors.blip_processors import BlipCaptionProcessor
from transformers import CLIPModel


class LavisModel(Backbone):
    pass


class BlipModel(LavisModel):
    def __init__(self, model_weights: str, label_texts: list[str], device="cpu"):
        super().__init__()
        self.model_weights = model_weights
        self.label_texts = label_texts
        self.device = device
        self.model, self.vision_processors, _ = load_model_and_preprocess("blip_feature_extractor",
                                                                          model_type=model_weights,
                                                                          is_eval=True,
                                                                          device=device)

    def forward(self, x: list):
        x = torch.cat([self.vision_processors["eval"](image).unsqueeze(0) for image in x])
        x = x.to(self.device)

        sample = {"image": x, "text_input": self.label_texts} # maybe we can have multiple samples somehow.

        image_features = self.model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
        text_features = self.model.extract_features(sample, mode="text").text_embeds_proj[:, 0]
        sims = (image_features @ text_features.t()) / self.model.temp
        return sims


class ALBEFModel(LavisModel):
    def __init__(self, model_weights: str, label_texts: list[str], device="cpu"):
        super().__init__()
        self.model_weights = model_weights
        self.label_texts = label_texts
        self.device = device
        self.model, self.vision_processors, _ = load_model_and_preprocess("albef_feature_extractor",
                                                                          model_type=model_weights,
                                                                          is_eval=True,
                                                                          device=device)

    def forward(self, x: list):
        x = torch.cat([self.vision_processors["eval"](image).unsqueeze(0) for image in x])
        x = x.to(self.device)

        sample = {"image": x, "text_input": self.label_texts} # maybe we can have multiple samples somehow.

        image_features = self.model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
        text_features = self.model.extract_features(sample, mode="text").text_embeds_proj[:, 0]
        sims = (image_features @ text_features.t()) / self.model.temp
        return sims


if __name__ == "__main__":
    b = BlipModel("base", ["a picture of a cat", "a picture of a dog"])