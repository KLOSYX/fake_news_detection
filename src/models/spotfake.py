from pathlib import Path

import torch
from einops import reduce
from torch import nn
from torchvision.models import VGG19_BN_Weights, vgg19_bn
from transformers import BertConfig, BertModel, get_constant_schedule_with_warmup

from src.models.components.fake_news_base import FakeNewsBase


class SpotFake(FakeNewsBase):
    """implement of SpotFake: http://arxiv.org/abs/2108.10509."""

    def __init__(
        self,
        lr=0.001,
        weight_decay=0.05,
        num_warmup_steps=400,
        dropout_prob=0.0,
        bert_name: str = "bert-base-chinese",
        pooler: str = "cls_token",
    ) -> None:
        assert pooler in [
            "cls_token",
            "avg_pool",
        ], "pooler must be one of [cls_token, avg_pool]"

        super().__init__()
        self.save_hyperparameters()
        bert_config = BertConfig.from_pretrained(bert_name, cache_dir=Path.home() / ".cache")
        # model
        self.bert = BertModel.from_pretrained(
            bert_name, cache_dir=Path.home() / ".cache", config=bert_config
        )
        self.vgg_model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)

        # layers without trainable weights
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # fcs
        self.img_fc1 = nn.Linear(self.vgg_model.classifier[6].in_features, 2742)
        self.img_fc2 = nn.Linear(2742, 32)
        self.text_fc1 = nn.Linear(self.bert.config.hidden_size, 32)
        self.fc = nn.Linear(64, 35)
        self.classifier = nn.Linear(35, 2)

        # pooler
        if pooler == "avg_pool":
            self.pool = nn.AdaptiveAvgPool1d(1)

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # modify the last fc layer of vgg
        new_classifier = nn.Sequential(*list(self.vgg_model.children())[-1][:6])
        self.vgg_model.classifier = new_classifier

        # freeze vgg
        self._freeze(self.vgg_model)

        # freeze bert
        self._freeze(self.bert)

    @staticmethod
    def _freeze(module):
        for n, p in module.named_parameters():
            p.requires_grad = False
            print("freeze", n)

    def forward(self, text_encodeds, img_encodeds):
        vgg_out = self.vgg_model(img_encodeds)
        if self.hparams.pooler == "pooler_output":
            bert_out = self.bert(**text_encodeds).pooler_output
        else:
            # bert_out = []
            attention_mask: torch.Tensor = text_encodeds.attention_mask  # [N, L]
            sequence_out: torch.Tensor = self.bert(**text_encodeds).last_hidden_state  # [N, L, d]
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
        x1 = self.img_fc1(vgg_out)
        x1 = self.dropout(x1)
        x1 = self.act(x1)
        x1 = self.img_fc2(x1)
        x1 = self.dropout(x1)
        x1 = self.act(x1)  # (N, 32)

        # text encoding
        x2 = self.text_fc1(bert_out)
        x2 = self.dropout(x2)
        x2 = self.act(x2)  # (N, 32)

        # multimodal repr
        x = torch.cat([x1, x2], dim=1)  # (N, 64)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.act(x)
        logits = self.classifier(x)  # (N, 2)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.hparams.num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
