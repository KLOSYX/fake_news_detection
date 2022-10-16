from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class FakeNewsBase(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        metrics = self._get_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _get_metrics(
        self,
    ):
        metric_collection = torchmetrics.MetricCollection(
            dict(
                accuracy=torchmetrics.classification.Accuracy(),
                fake_precision=torchmetrics.classification.Precision(
                    ignore_index=0, multiclass=True, num_classes=2
                ),
                fake_recall=torchmetrics.classification.Recall(
                    ignore_index=0, multiclass=True, num_classes=2
                ),
                fake_f1score=torchmetrics.classification.F1Score(
                    ignore_index=0, multiclass=True, num_classes=2
                ),
                real_precision=torchmetrics.classification.Precision(
                    ignore_index=1, multiclass=True, num_classes=2
                ),
                real_recall=torchmetrics.classification.Recall(
                    ignore_index=1, multiclass=True, num_classes=2
                ),
                real_f1score=torchmetrics.classification.F1Score(
                    ignore_index=1, multiclass=True, num_classes=2
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

    def forward_loss(self, batch):
        text_encodeds, img_encodeds, labels = batch
        logits = self(text_encodeds, img_encodeds)  # [N]
        loss = self.criterion(logits, labels.to(torch.float))
        return torch.sigmoid(logits), labels, loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        logits, labels, loss = self.forward_loss(batch)
        self.log_dict({"train/loss": loss})
        train_metrics_dict = self.train_metrics(logits, labels)
        self.log_dict(train_metrics_dict)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        logits, labels, loss = self.forward_loss(batch)
        self.log_dict({"val/loss": loss})
        return (logits, labels)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.val_metrics.reset()
        logits, labels = zip(*outputs)
        val_metrics_dict = self.val_metrics(torch.cat(logits), torch.cat(labels))
        self.log_dict(val_metrics_dict, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        logits, labels, loss = self.forward_loss(batch)
        self.log_dict({"test/loss": loss})
        return (logits, labels)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.test_metrics.reset()
        logits, labels = zip(*outputs)
        test_metrics_dict = self.test_metrics(torch.cat(logits), torch.cat(labels))
        self.log_dict(test_metrics_dict)
