import pytorch_lightning as pl
import torch
from torch import nn
from transformers import BertConfig

from src.models.components.sim_bert import SimBertModel


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, **kwargs):
        pass


class TrainSimBert(pl.LightningModule):
    def __init__(
        self,
        bert_name: str = "hfl/chinese-roberta-wwm-ext",
        vector_dim: int = 512,
    ) -> None:
        super().__init__()
        # model
        config = BertConfig.from_pretrained(bert_name, cache_dir="/data/.cache")
        config.vector_dim = vector_dim
        self.model = SimBertModel.from_pretrained(
            bert_name, config=config, cache_dir="/data/.cache"
        )
        # loss
