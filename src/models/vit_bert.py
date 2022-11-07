from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertConfig, ViTModel, get_constant_schedule_with_warmup

from src.models.components.fake_news_base import FakeNewsBase
from src.models.components.med import BertModel


class VitBert(FakeNewsBase):
    def __init__(
        self,
        visual_encoder: str = "google/vit-base-patch16-224-in21k",
        text_encoder: str = "bert-base-uncased",
        learning_rate: float = 0.001,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 400,
        fc_hidden_size: int = 256,
        fc_dropout_prob: float = 0.4,
    ) -> None:
        super().__init__()
        self.visual_encoder = ViTModel.from_pretrained(
            visual_encoder, cache_dir=Path.home() / ".cache"
        )

        self.text_decoder_config = BertConfig.from_pretrained(
            text_encoder, cache_dir=Path.home() / ".cache"
        )
        # add cross attention layers to interact with visual encoder
        self.text_decoder_config.is_decoder = True
        self.text_decoder_config.add_cross_attention = True
        self.text_decoder_config.encoder_width = self.visual_encoder.config.hidden_size
        self.text_decoder = BertModel.from_pretrained(
            pretrained_model_name_or_path=text_encoder,
            config=self.text_decoder_config,
            cache_dir=Path.home() / ".cache",
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.text_decoder_config.hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout_prob),
            nn.Linear(fc_hidden_size, 2),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        self._freeze(self.visual_encoder)
        self._freeze(self.text_decoder)

    def forward(self, text_encodes, img_encodes):
        # loss for mlm
        encoder_outputs = self.visual_encoder(**img_encodes, return_dict=True)
        last_visual_tokens: torch.Tensor = encoder_outputs.last_hidden_state  # (N, L, dim)
        encoder_attention_mask = torch.ones(
            last_visual_tokens.size(0), last_visual_tokens.size(1)
        ).to(last_visual_tokens.device)
        outputs = self.text_decoder(
            input_ids=text_encodes.input_ids,
            attention_mask=text_encodes.attention_mask,
            encoder_hidden_states=last_visual_tokens,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
            output_hidden_states=True,
            mode="multimodal",
        )

        model_out = outputs.pooler_output
        logits = self.classifier(model_out)

        return logits

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# debug
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "vit_bert.yaml")
    dm_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "weibo.yaml")
    with omegaconf.open_dict(dm_cfg):
        dm_cfg["tokenizer_name"] = "bert-base-uncased"
        dm_cfg["processor_name"] = "google/vit-base-patch16-224-in21k"
        dm_cfg["max_length"] = 200
        dm_cfg["img_path"] = root / "data" / "MM17-WeiboRumorSet/images_filtered"
        dm_cfg["train_path"] = root / "data" / "MM17-WeiboRumorSet/train_data.json"
        dm_cfg["test_path"] = root / "data" / "MM17-WeiboRumorSet/test_data.json"
        dm_cfg["vis_model_type"] = "other"

    model = hydra.utils.instantiate(model_cfg)
    dm = hydra.utils.instantiate(dm_cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    model.forward_loss(batch)
    pass
