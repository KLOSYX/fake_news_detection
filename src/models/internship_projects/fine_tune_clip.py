# !/usr/bin/env python3
# @Time : 2022/8/19
# @Author : anbinx
# @Email : klosyx@outlook.com
# @Model Description: Fine tune CLIP model.

from typing import Any, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
from transformers import (
    BertForSequenceClassification,
    CLIPModel,
    get_constant_schedule_with_warmup,
)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        labels: torch.Tensor,
    ):
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2


class CLIP(pl.LightningModule):
    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        text_encoder_name: str = "IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
        lr: float = 0.0001,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.clip_model = CLIPModel.from_pretrained(clip_name, cache_dir="~/.cache")
        self.text_encoder = BertForSequenceClassification.from_pretrained(
            text_encoder_name, cache_dir="~/.cache"
        )
        self.criterion = Loss()
        self.val_best = torchmetrics.MinMetric()

        self._freeze_except_startwith(self.clip_model, [])
        # self._freeze_except_startwith(self.text_encoder, ["bert.embeddings"])

    def _freeze_except_startwith(self, module: Any, except_list: list, verbose=True) -> None:
        for n, p in module.named_parameters():
            for name in except_list:
                if n.startswith(name):
                    if verbose:
                        print(f"not freezing {n}")
                    break
            else:
                p.requires_grad = False
                if verbose:
                    print(f"freeze {n}")

    def encode_image(self, img_encoded: Any) -> Any:
        image_features = self.clip_model.get_image_features(**img_encoded)
        return image_features

    def encode_text(self, text_encoded: Any) -> Any:
        text_features = self.text_encoder(**text_encoded).logits
        return text_features

    def forward(self, text_encoded: Any, image_encoded: Any) -> Any:
        image_features = self.encode_image(image_encoded)
        text_features = self.encode_text(text_encoded)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # scale the text_embedding
        logit_scale = self.clip_model.logit_scale.exp()
        return image_features, logit_scale * text_features

    def forward_loss(self, batch: Any) -> torch.Tensor:
        text_encoded, image_encoded = batch
        img_embed, scale_text_embed = self(text_encoded, image_encoded)  # [local batch, C]
        hidden_size = img_embed.size(1)
        all_img = self.all_gather(img_embed, sync_grads=True).reshape(
            -1, hidden_size
        )  # [global batch, C]
        all_scale_text_embed = self.all_gather(scale_text_embed, sync_grads=True).reshape(
            -1, hidden_size
        )  # [global batch, C]

        logits_per_image = all_img @ all_scale_text_embed.t()  # [global batch, global batch]
        logits_per_text = all_scale_text_embed @ all_img.t()  # [global batch, global batch]

        ground_truth = torch.arange(logits_per_image.shape[0]).to(self.device)
        loss = self.criterion(logits_per_image, logits_per_text, ground_truth)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.forward_loss(batch)
        self.log("train/loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss = self.forward_loss(batch)
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        loss_epoch = torch.stack(outputs).mean()
        self.val_best.update(loss_epoch.item())
        self.log("val/best", self.val_best.compute(), sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss = self.forward_loss(batch)
        self.log("test/loss", loss, sync_dist=True)


if __name__ == "__main__":
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    from src.datamodules.multi_modal_data import MultiModalDatamodule

    model = CLIP()
    datamodules = MultiModalDatamodule(
        train_path="/data/clean_raw_text/all_data_cleaned_wk_refine.json",
        tokenizer_name="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
        processor_name="openai/clip-vit-base-patch32",
    )
    datamodules.setup("fit")
    data = iter(datamodules.train_dataloader())
    while True:
        text_encoded, image_encoded = next(data)
        # model(text_encoded, image_encoded)
        model.training_step((text_encoded, image_encoded), 0)
