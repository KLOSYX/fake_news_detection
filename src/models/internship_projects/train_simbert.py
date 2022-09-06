from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from transformers import BertConfig, get_constant_schedule_with_warmup

from src.models.components.sim_bert import SimBertModel


class SimilarityLoss(nn.Module):
    def __init__(self, logit_scale: float = 30.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        assert 0.0 <= label_smoothing <= 1.0
        self.logit_scale = logit_scale  # according to: https://spaces.ac.cn/archives/7427
        self.label_smoothing = label_smoothing

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        batch_size = vectors.size(0)
        labels = torch.arange(batch_size, device=vectors.device)
        labels[::2] = labels[::2] + 1
        labels[1::2] = labels[1::2] - 1
        norm_vector = F.normalize(vectors, dim=1)  # (N, vector_dim)
        similarity = torch.matmul(norm_vector, norm_vector.t()) * self.logit_scale  # (N, N)
        similarity.fill_diagonal_(torch.finfo(similarity.dtype).min)
        similarity = F.softmax(similarity, dim=1)
        loss = F.cross_entropy(similarity, labels, label_smoothing=self.label_smoothing)
        return loss


class TrainSimBert(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        bert_name: str = "hfl/chinese-roberta-wwm-ext",
        vector_dim: int = 512,
        logit_scale: float = 30.0,
        label_smoothing: float = 0.0,
        num_warmup_steps: int = 4000,
    ) -> None:
        super().__init__()
        # model
        config = BertConfig.from_pretrained(bert_name, cache_dir="~/.cache")
        config.vector_dim = vector_dim
        self.model = SimBertModel.from_pretrained(bert_name, config=config, cache_dir="~/.cache")
        # loss
        self.criterion = SimilarityLoss(logit_scale=logit_scale, label_smoothing=label_smoothing)
        # hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

    def forward(self, inputs: Dict) -> torch.Tensor:
        outputs, vector = self.model.forward_vector(**inputs)
        gen_loss = outputs.loss
        all_vector = self.all_gather(vector, sync_grads=True)
        if len(all_vector.shape) > 2:  # multi-gpu, global_size in dim 0
            all_vector = torch.cat([x for x in all_vector], dim=0)
        sim_loss = self.criterion(all_vector)
        return gen_loss, sim_loss

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
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

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gen_loss, sim_loss = self.forward(batch)
        loss = gen_loss + sim_loss
        self.log_dict(
            {
                "train/gen_loss": gen_loss,
                "train/sim_loss": sim_loss,
                "train/loss": loss,
            },
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        gen_loss, sim_loss = self.forward(batch)
        loss = gen_loss + sim_loss
        self.log_dict(
            {"val/gen_loss": gen_loss, "val/sim_loss": sim_loss, "val/loss": loss},
            sync_dist=True,
        )


if __name__ == "__main__":
    # debug
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "sim_bert_model.yaml")
    model = hydra.utils.instantiate(model_cfg)
    dm_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "sim_bert_data.yaml")
    dm = hydra.utils.instantiate(dm_cfg)
    dm.setup("fit")
    data = iter(dm.train_dataloader())
    while True:
        loss = model.training_step(next(data), 0)
