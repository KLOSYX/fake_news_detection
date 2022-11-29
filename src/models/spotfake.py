from pathlib import Path

import torch
from einops import reduce
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    VGG19_BN_Weights,
    mobilenet_v3_large,
    vgg19_bn,
)
from transformers import AutoModel, BertModel, get_constant_schedule_with_warmup

from src.datamodules.fake_news_data import FakeNewsItem
from src.models.components.fake_news_base import FakeNewsBase


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()

        # modify the last fc layer of vgg
        self.vgg_model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        new_classifier = nn.Sequential(*list(self.vgg_model.children())[-1][:6])

        self.fc = nn.Linear(self.vgg_model.classifier[6].in_features, 2742)
        self.vgg_model.classifier = new_classifier

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.vgg_model(x)
        x = self.fc(x)
        x = self.act(x)
        return x  # (batch_size, 2742)


class MobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.mobilenetv3 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        new_classifier = nn.Sequential(*list(self.mobilenetv3.children())[-1][:3])

        self.fc = nn.Linear(self.mobilenetv3.classifier[3].in_features, 2742)
        self.mobilenetv3.classifier = new_classifier

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.mobilenetv3(x)
        x = self.fc(x)
        x = self.act(x)
        return x  # (batch_size, 2742)


class SpotFake(FakeNewsBase):
    """implement of SpotFake: http://arxiv.org/abs/2108.10509."""

    def __init__(
        self,
        lr=0.001,
        weight_decay=0.05,
        num_warmup_steps=400,
        dropout_prob=0.0,
        bert_name: str = "bert-base-chinese",
        visual_encoder: nn.Module = VGG19(),
        pooler: str = "cls_token",
    ) -> None:
        assert pooler in [
            "cls_token",
            "avg_pool",
        ], "pooler must be one of [cls_token, avg_pool]"

        super().__init__()

        # hyper parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.pooler = pooler

        # bert_config = BertConfig.from_pretrained(bert_name, cache_dir=Path.home() / ".cache")
        # model
        self.bert = AutoModel.from_pretrained(bert_name, cache_dir=Path.home() / ".cache")
        self.visual_encoder = visual_encoder

        # layers without trainable weights
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # fcs
        self.img_encoder = nn.Sequential(
            self.dropout,
            nn.Linear(2742, 32),
            self.act,
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 768),
            self.act,
            self.dropout,
            nn.Linear(768, 32),
            self.act,
        )
        self.fc = nn.Linear(64, 35)
        self.classifier = nn.Linear(35, 2)

        # pooler
        # if pooler == "avg_pool":
        #     self.pool = nn.AdaptiveAvgPool1d(1)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # freeze ve
        self._freeze(self.visual_encoder)

        # freeze bert
        self._freeze(self.bert)

    def forward(self, item: FakeNewsItem) -> torch.Tensor:
        visual_out = self.visual_encoder(item.image_encoded)  # (batch_size, 2742)
        if self.pooler == "pooler_output":
            bert_out = self.bert(**item.text_encoded).pooler_output
        else:
            # bert_out = []
            attention_mask: torch.Tensor = item.text_encoded.attention_mask  # [N, L]
            sequence_out: torch.Tensor = self.bert(
                **item.text_encoded
            ).last_hidden_state  # [N, L, d]
            input_mask_expanded: torch.Tensor = (
                attention_mask.unsqueeze(-1).expand(sequence_out.size()).float()
            )
            t = sequence_out * input_mask_expanded  # [N, L, d]
            sum_embed = reduce(t, "N L d -> N d", "sum")
            sum_mask = reduce(input_mask_expanded, "N L d -> N d", "sum")
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # make sure not divided by zero
            # bert_out.append(sum_embed / sum_mask)
            # bert_out = torch.cat(bert_out, dim=1)
            bert_out = sum_embed / sum_mask

        # visual encoding
        x1 = self.img_encoder(visual_out)
        # x1 = self.dropout(x1)  # (N, 32)

        # text encoding
        x2 = self.text_encoder(bert_out)
        # x2 = self.dropout(x2)  # (N, 32)

        # multimodal repr
        x = torch.cat([x1, x2], dim=1)  # (N, 64)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.act(x)
        logits = self.classifier(x)  # (N, 2)
        # logits = logits.squeeze(-1)  # (N,)

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
