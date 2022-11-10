from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import get_constant_schedule_with_warmup

from src.models.components.blip_base import blip_feature_extractor
from src.models.components.fake_news_base import FakeNewsBase


class TextCNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 300,
        conv_out_channels: int = 128,
        kernel_sizes: Tuple[int] = (3, 4, 5),
        dropout_prob: float = 0.0,
        hidden_size: int = 64,
        output_size: int = 768,
    ):
        super().__init__()
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(1, conv_out_channels, (k, embedding_dim), dtype=torch.float)
                for k in kernel_sizes
            ]
        )
        self.act = nn.ReLU()
        self.proj = nn.Sequential(
            nn.Linear(len(kernel_sizes) * conv_out_channels, hidden_size),
            self.act,
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size),
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (N, L, E)
        x = [self.act(conv(x.unsqueeze(1))) for conv in self.conv]  # [(N, Co, L', 1), ...]*len(Ks)
        x = [self.max_pool(i.squeeze(-1)).squeeze(-1) for i in x]  # [(N, Co)]*len(Ks)
        x = torch.cat(x, 1)  # (N, len(Ks)*Co)
        x = self.proj(x)  # (N, output_size)
        return x


class LstmEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_prob: float = 0.4,
        out_size: int = 768,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.proj = (
            nn.Linear(hidden_size * 2, out_size)
            if bidirectional
            else nn.Linear(hidden_size, out_size)
        )
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        # x: (N, L, E)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        if self.bidirectional:
            out = self.proj(torch.cat([h[-2], h[-1]], dim=1))
        else:
            out = self.proj(h[-1])
        return out


class BlipKb(FakeNewsBase):
    def __init__(
        self,
        model_path: str,
        med_config: str,
        kb_encoder: nn.Module,
        fc_hidden_size: int = 256,
        dropout_prob: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 400,
        label_smoothing: float = 0.1,
        is_freeze_blip: bool = True,
        fine_tune_visual_encoder: bool = False,
        fine_tune_text_encoder: bool = False,
    ):
        super().__init__()
        # hyper parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        # models
        self.blip = blip_feature_extractor(
            pretrained=model_path, med_config=med_config, image_size=224, vit="base"
        )
        self.kb_encoder = kb_encoder
        self.classifier = nn.Sequential(
            nn.Linear(768, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_size, 2),
        )
        self.register_buffer("null_entities", torch.zeros(1, 768))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if is_freeze_blip:
            self._freeze(self.blip)
            if fine_tune_visual_encoder:
                for n, p in self.blip.visual_encoder.named_parameters():
                    p.requires_grad = True
            if fine_tune_text_encoder:
                for n, p in self.blip.text_encoder.named_parameters():
                    p.requires_grad = True

    def forward(self, text_encodeds, img_encodeds):
        # encode images
        image_embeds = self.blip.visual_encoder(img_encodeds)
        b, k = image_embeds.shape[:2]
        s = text_encodeds.input_ids.shape[1]
        image_atts = torch.ones((b, s, k), dtype=torch.long).to(img_encodeds.device)

        # encode knowledge
        kb_embeds = text_encodeds["kb_embeddings"].to(img_encodeds.device).to(torch.float32)
        kb_true_annotation_nums = text_encodeds["kb_true_annotation_nums"]
        kb_true_desc_lengths = text_encodeds["kb_true_desc_lengths"]
        kb_atts = text_encodeds["kb_cross_atts"]  # (b, 32, s)

        if isinstance(self.kb_encoder, LstmEncoder):
            kb_out = self.kb_encoder(kb_embeds, kb_true_desc_lengths.detach().cpu())
        else:
            kb_out = self.kb_encoder(kb_embeds)  # (total_kb, 768)

        # Split kb_out
        kb_embeds = []
        start = 0
        for n in kb_true_annotation_nums:
            end = start + n
            if start == end:
                kb_embeds.append(self.null_entities)
            else:
                kb_embeds.append(kb_out[start:end])
            start = end
        kb_embeds = pad_sequence(
            kb_embeds, batch_first=True, padding_value=0
        )  # (N, max_length, 768)
        kb_atts = kb_atts[:, :, : kb_embeds.size(1)]  # (N, max_length)

        image_kb_embeds = torch.cat([image_embeds, kb_embeds], dim=1)
        image_kb_atts = torch.cat([image_atts, kb_atts], dim=-1)  # (b, s, l + max_length)

        text_encodeds.input_ids[:, 0] = self.blip.tokenizer.enc_token_id

        output = self.blip.text_encoder(
            input_ids=text_encodeds["input_ids"],
            attention_mask=text_encodeds["attention_mask"],
            encoder_hidden_states=image_kb_embeds,
            encoder_attention_mask=image_kb_atts,
            return_dict=True,
        )

        logits = self.classifier(output.last_hidden_state[:, 0, :])
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
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
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "blip_kb.yaml")
    dm_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "weibo.yaml")
    with omegaconf.open_dict(dm_cfg):
        model_cfg["_target_"] = "src.models.blip_kb.BlipKb"
        dm_cfg[
            "train_path"
        ] = "/home/anbinx/develop/notebooks/wiki_entity_link/weibo_train_data_wiki.json"
        dm_cfg["w2v_path"] = "/home/anbinx/develop/notebooks/wiki_entity_link/wiki_desc_vec.npy"
        dm_cfg["img_path"] = root / "data" / "MM17-WeiboRumorSet" / "images_filtered"
        dm_cfg[
            "test_path"
        ] = "/home/anbinx/develop/notebooks/wiki_entity_link/weibo_test_data_wiki.json"
        dm_cfg["dataset_name"] = "weibo_kb"
        dm_cfg["batch_size"] = 2

    model = hydra.utils.instantiate(model_cfg)
    dm = hydra.utils.instantiate(dm_cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    model.forward_loss(batch)
    pass
