from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops import reduce
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import BertConfig, BertModel, get_constant_schedule_with_warmup

from utils.loss.focal_loss import FocalLoss


class BertSequenceClassification(pl.LightningModule):
    def __init__(
        self,
        bert_name: str = "hfl/chinese-roberta-wwm-ext",
        num_classes: int = 10,
        focal_loss: bool = False,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.05,
        dropout_prob: float = 0.1,
        label_smooth: bool = False,
        num_warmup_steps: int = 0,
        pooler: str = "first_token",
    ) -> None:
        assert pooler in [
            "first_token",
            "average",
        ], "pooler must be either 'first_token' or 'average'"

        super().__init__()
        self.save_hyperparameters()

        # model
        config = BertConfig.from_pretrained(bert_name, cache_dir="/data/.cache")
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        config.num_labels = num_classes
        self.bert = BertModel.from_pretrained(bert_name, cache_dir="/data/.cache", config=config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        # loss function
        self.criterion = (
            nn.CrossEntropyLoss(label_smoothing=0.1 if label_smooth else 0)
            if not focal_loss
            else FocalLoss(alpha=[1] * num_classes, num_classes=num_classes)
        )

        # metric
        self.train_acc = self.get_top_k_metric_collection(
            "Accuracy", prefix="train", num_classes=num_classes, multiclass=True
        )
        # self.train_f1_score = self.get_top_k_metric_collection('F1Score', prefix='train', num_classes=num_classes, average='macro', multiclass=True)
        self.val_acc = self.get_top_k_metric_collection(
            "Accuracy", prefix="val", num_classes=num_classes, multiclass=True
        )
        # self.val_f1_score = self.get_top_k_metric_collection('F1Score', prefix='val', num_classes=num_classes, average='macro', multiclass=True)
        self.test_acc = self.get_top_k_metric_collection(
            "Accuracy", prefix="test", num_classes=num_classes, multiclass=True
        )
        self.val_best = torchmetrics.MaxMetric()

    def get_top_k_metric_collection(
        self, metric="Accuracy", prefix="train", num_classes=None, top_k=3, **kwargs
    ):
        return torchmetrics.MetricCollection(
            {
                f"{prefix}/{metric.lower()}_top_{k}": getattr(torchmetrics, metric)(
                    num_classes=num_classes, top_k=k, **kwargs
                )
                for k in range(1, top_k + 1)
            }
        )

    def forward(self, tokens):
        if self.hparams.pooler == "first_token":
            bert_out = self.bert(**tokens).pooler_output
        else:
            bert_out = []
            attention_mask: torch.Tensor = tokens.attention_mask  # [N, L]
            sequence_out: torch.Tensor = self.bert(**tokens).last_hidden_state  # [N, L, d]
            input_mask_expanded: torch.Tensor = (
                attention_mask.unsqueeze(-1).expand(sequence_out.size()).to(sequence_out.dtype)
            )
            t = sequence_out * input_mask_expanded  # [N, L, d]
            sum_embed = reduce(t, "N L d -> N d", "sum")
            sum_mask = reduce(input_mask_expanded, "N L d -> N d", "sum")
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # make sure not divided by zero
            bert_out.append(sum_embed / sum_mask)
            bert_out = torch.cat(bert_out, dim=1)
        logits = self.classifier(bert_out)
        return logits

    def forward_loss(self, tokens, class_ids):
        logits = self(tokens)
        labels = F.one_hot(class_ids, num_classes=self.hparams.num_classes).to(torch.float)
        loss = self.criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_start(self) -> None:
        self.train_acc.reset()
        self.val_acc.reset()
        self.val_best.reset()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        tokens, class_ids = batch
        logits = self(tokens)
        loss = self.forward_loss(tokens, class_ids)
        self.log_dict({"train/loss": loss}, sync_dist=True)
        self.log_dict(self.train_acc(logits, class_ids), sync_dist=True)
        # self.log_dict(self.train_f1_score(logits, labels))
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        tokens, class_ids = batch
        logits = self(tokens)
        loss = self.forward_loss(tokens, class_ids)
        self.log_dict({"val/loss": loss}, sync_dist=True)
        self.log_dict(self.val_acc(logits, class_ids), sync_dist=True)
        # self.log_dict(self.val_f1_score(logits, labels))

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.val_best.update(self.val_acc.compute()["val/accuracy_top_1"])
        self.val_acc.reset()
        self.log("val/best_acc", self.val_best.compute(), sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        tokens, class_ids = batch
        logits = self(tokens)
        loss = self.forward_loss(tokens, class_ids)
        self.log_dict({"test/loss": loss}, sync_dist=True)
        self.log_dict(self.test_acc(logits, class_ids), sync_dist=True)
        # self.log_dict(self.val_f1_score(logits, labels))

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.test_acc.reset()
