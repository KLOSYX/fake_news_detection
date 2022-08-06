import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertForSequenceClassification, CLIPModel


class ClipBase(nn.Module):
    def __init__(
        self,
        clip_name="openai/clip-vit-base-patch32",
        bert_name="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
    ) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name, cache_dir="/data/.cache")
        self.bert = BertForSequenceClassification.from_pretrained(
            bert_name, cache_dir="/data/.cache"
        )

    def forward(self, text_encoded, img_encoded):
        img_features = self.clip.get_image_features(**img_encoded)
        text_features = self.bert(**text_encoded).logits
        # norm
        # img_features = F.normalize(img_features, dim=1)
        # text_features = F.normalize(text_features, dim=1)
        return img_features, text_features

    def freeze_clip(self):
        for k, v in self.named_parameters():
            if k.startswith("clip"):
                v.requires_grad = False
                print("freeze", k)

    def freeze_bert(self):
        for k, v in self.named_parameters():
            if k.startswith("bert"):
                v.requires_grad = False
                print("freeze", k)
