import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT, Optional
from transformers import BertForMaskedLM, get_constant_schedule_with_warmup

transformers.logging.set_verbosity_error()
import torchmetrics


class PureTextMlmPretrain(pl.LightningModule):
    def __init__(
        self,
        bert_name: str = "hfl/chinese-macbert-base",
        num_warmup_steps: int = 0,
        lr: float = 0.0001,
    ) -> None:
        super().__init__()
        # model
        self.model = BertForMaskedLM.from_pretrained(bert_name, cache_dir="~/.cache")
        # metrics
        self.train_acc = torchmetrics.Accuracy(ignore_index=-100)
        self.val_acc = torchmetrics.Accuracy(ignore_index=-100)
        # hyperparameters
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps

    def forward(self, encoded):
        outputs = self.model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            labels=encoded.labels,
            return_dict=True,
        )
        return outputs.loss, outputs.logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        tokens = batch
        loss, logits = self(tokens)
        predictions = torch.argmax(logits, dim=-1)  # (N, L)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.train_acc(predictions, tokens.labels),
            },
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        tokens = batch
        loss, logits = self(tokens)
        predictions = torch.argmax(logits, dim=-1)  # (N, L)
        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_acc(predictions, tokens.labels)},
            sync_dist=True,
        )
