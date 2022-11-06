from pathlib import Path

import torch
import transformers
from einops import reduce
from torch import nn

from src.models.components.fake_news_base import FakeNewsBase


class ViLT(FakeNewsBase):
    def __init__(
        self,
        pooler: str = "cls_token",
        learning_rate: float = 1e-5,
        weight_decay: float = 0.05,
        model_name: str = "dandelin/vilt-b32-mlm",
        fc_hidden_size: int = 256,
        fc_dropout_prob: float = 0.0,
        num_warmup_steps: int = 200,
        freeze_backbone: bool = True,
        fine_tune_last_n_layers: int = 0,
    ):
        super().__init__()

        # hperparameters
        assert pooler in ["cls_token", "avg_pool"], "pooler must be one of [cls_token, avg_pool]"
        self.pooler = pooler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fc_dropout_prob = fc_dropout_prob
        self.num_warmup_steps = num_warmup_steps

        # models
        self.model = transformers.ViltModel.from_pretrained(
            model_name, cache_dir=Path.home() / ".cache"
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, fc_hidden_size),
            nn.GELU(),
            nn.Dropout(self.fc_dropout_prob),
            nn.Linear(fc_hidden_size, 2),
        )
        if freeze_backbone:
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            if fine_tune_last_n_layers > 0:
                assert (
                    0 <= fine_tune_last_n_layers <= 11
                ), "fine_tune_last_n_layers must be in [0, 11]"
                for _, p in self.model.encoder.layer[-fine_tune_last_n_layers:].named_parameters():
                    p.requires_grad = True

        # criterion
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_encodeds, img_encodeds):
        text_encodeds.update(img_encodeds)
        outputs = self.model(
            **text_encodeds,
        )

        if self.pooler == "cls_token":
            bert_out = outputs.pooler_output
        else:
            attention_mask: torch.Tensor = torch.concat(
                [text_encodeds["attention_mask"], img_encodeds["pixel_mask"]], dim=1
            )
            sequence_out: torch.Tensor = outputs.last_hidden_state  # [N, L, d]
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

        logits = self.classifier(bert_out)

        return logits

    def configure_optimizers(self):
        param_optimizer = list((n, p) for n, p in self.named_parameters() if p.requires_grad)
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
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# debug
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "vilt.yaml")
    dm_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "weibo.yaml")
    with omegaconf.open_dict(dm_cfg):
        dm_cfg["tokenizer_name"] = "dandelin/vilt-b32-mlm"
        dm_cfg["processor_name"] = "dandelin/vilt-b32-mlm"
        dm_cfg["max_length"] = 40
        dm_cfg["img_path"] = root / "data" / "MM17-WeiboRumorSet/images_filtered"
        dm_cfg["train_path"] = root / "data" / "MM17-WeiboRumorSet/train_data.json"
        dm_cfg["test_path"] = root / "data" / "MM17-WeiboRumorSet/test_data.json"
        dm_cfg["vis_model_type"] = "vilt"

    model = hydra.utils.instantiate(model_cfg)
    dm = hydra.utils.instantiate(dm_cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    model.forward_loss(batch)
    pass
