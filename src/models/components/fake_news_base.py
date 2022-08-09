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
                accuracy=torchmetrics.Accuracy(),
                fake_precision=torchmetrics.Precision(ignore_index=0),
                fake_recall=torchmetrics.Recall(ignore_index=0),
                fake_f1score=torchmetrics.F1Score(ignore_index=0),
                real_precision=torchmetrics.Precision(ignore_index=1),
                real_recall=torchmetrics.Recall(ignore_index=1),
                real_f1score=torchmetrics.F1Score(ignore_index=1),
            ),
        )
        return metric_collection

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implememnted!")

    def configure_optimizers(self):
        raise NotImplementedError("Configure_optimizers method not implemented!")

    def on_train_start(self):
        assert hasattr(self, "criterion"), "criterion not found"

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        text_encodeds, img_encodeds, labels = batch
        logits = self(text_encodeds, img_encodeds)  # [N, 2]
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.log_dict({"train/loss": loss})
        self.log_dict(self.train_metrics(preds, labels))
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        text_encodeds, img_encodeds, labels = batch
        logits = self(text_encodeds, img_encodeds)  # [N, 2]
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.log_dict({"val/loss": loss})
        self.log_dict(self.val_metrics(preds, labels))
        return (preds, labels)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.val_metrics.reset()
        preds, labels = zip(*outputs)
        self.log_dict(self.val_metrics(torch.cat(preds), torch.cat(labels)))

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        text_encodeds, img_encodeds, labels = batch
        logits = self(text_encodeds, img_encodeds)  # [N, 2]
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)  # [N]
        self.log_dict({"test/loss": loss}, on_step=False, on_epoch=True)
        return (preds, labels)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.test_metrics.reset()
        preds, labels = zip(*outputs)
        self.log_dict(self.test_metrics(torch.cat(preds), torch.cat(labels)))
