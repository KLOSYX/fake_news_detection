import torch
from torch import nn
from transformers import get_constant_schedule_with_warmup

from src.models.components.blip_base import blip_feature_extractor
from src.models.components.fake_news_base import FakeNewsBase


class SimpleBlip(FakeNewsBase):
    def __init__(
        self,
        model_path: str,
        med_config: str,
        fc_hidden_size: int = 256,
        dropout_prob: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 400,
        is_freeze_blip: bool = True,
        fine_tune_visual_encoder: bool = False,
        fine_tune_text_encoder: bool = False,
    ):
        super().__init__()
        # hyper parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        # models
        self.blip = blip_feature_extractor(
            pretrained=model_path, med_config=med_config, image_size=224, vit="base"
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_size, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

        if is_freeze_blip:
            self._freeze(self.blip)
            if fine_tune_visual_encoder:
                for n, p in self.blip.visual_encoder.named_parameters():
                    p.requires_grad = True
            if fine_tune_text_encoder:
                for n, p in self.blip.text_encoder.named_parameters():
                    p.requires_grad = True

    def forward(self, text_encodeds, img_encodeds):
        mm_features = self.blip(img_encodeds, text_encodeds, mode="multimodal")[
            :, 0, :
        ]  # (N, 768)
        logits = self.classifier(mm_features)
        return logits

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(n_d in n for n_d in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(n_d in n for n_d in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
