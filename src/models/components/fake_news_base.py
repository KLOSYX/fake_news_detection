from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from src.datamodules.fake_news_data import FakeNewsItem


class FakeNewsBase(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        metrics = self._get_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.val_best = torchmetrics.MaxMetric()

    def _get_metrics(
        self,
    ):
        metric_collection = torchmetrics.MetricCollection(
            dict(
                accuracy=torchmetrics.classification.Accuracy(task="multiclass", num_classes=2),
                fake_precision=torchmetrics.classification.Precision(
                    task="multiclass", ignore_index=0, num_classes=2
                ),
                fake_recall=torchmetrics.classification.Recall(
                    task="multiclass", ignore_index=0, num_classes=2
                ),
                fake_f1score=torchmetrics.classification.F1Score(
                    task="multiclass", ignore_index=0, num_classes=2
                ),
                real_precision=torchmetrics.classification.Precision(
                    task="multiclass", ignore_index=1, num_classes=2
                ),
                real_recall=torchmetrics.classification.Recall(
                    task="multiclass", ignore_index=1, num_classes=2
                ),
                real_f1score=torchmetrics.classification.F1Score(
                    task="multiclass", ignore_index=1, num_classes=2
                ),
            ),
        )
        return metric_collection

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implememnted!")

    def configure_optimizers(self):
        raise NotImplementedError("Configure_optimizers method not implemented!")

    def on_train_start(self):
        assert hasattr(self, "criterion"), "criterion not found"
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_best.reset()

    def forward_loss(self, batch: FakeNewsItem) -> tuple:
        # text_encodeds, img_encodeds, labels = batch
        logits = self(batch)  # [N, 2]
        loss = self.criterion(logits, batch.label)
        return logits, batch.label, loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        logits, labels, loss = self.forward_loss(batch)
        preds = torch.argmax(logits, dim=1)  # (N,)
        self.log_dict({"train/loss": loss})
        train_metrics_dict = self.train_metrics(preds, labels)
        self.log_dict(train_metrics_dict)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        logits, labels, loss = self.forward_loss(batch)
        self.log_dict({"val/loss": loss})
        return (logits, labels)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]) -> None:
        self.val_metrics.reset()
        logits, labels = zip(*outputs)
        preds = torch.argmax(torch.cat(logits), dim=1)
        val_metrics_dict = self.val_metrics(preds, torch.cat(labels))
        self.val_best(val_metrics_dict["val/accuracy"])
        self.log_dict(
            val_metrics_dict, sync_dist=True, on_epoch=True, on_step=False, prog_bar=True
        )
        self.log(
            "val_best",
            self.val_best.compute(),
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        logits, labels, loss = self.forward_loss(batch)
        self.log_dict({"test/loss": loss})
        return (logits, labels)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]) -> None:
        self.test_metrics.reset()
        logits, labels = zip(*outputs)
        preds = torch.argmax(torch.cat(logits), dim=1)
        test_metrics_dict = self.test_metrics(preds, torch.cat(labels))
        self.log_dict(test_metrics_dict)

    @staticmethod
    def _freeze(module, verbose=False):
        for n, p in module.named_parameters():
            p.requires_grad = False
            if verbose:
                print("freeze", n)
