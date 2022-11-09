from itertools import zip_longest
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
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


class BlipKb(FakeNewsBase):
    def __init__(
        self,
        model_path: str,
        med_config: str,
        fc_hidden_size: int = 256,
        dropout_prob: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 400,
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
        self.text_cnn = TextCNN()
        self.classifier = nn.Sequential(
            nn.Linear(768, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_size, 2),
        )
        self.null_entities = nn.Parameter(torch.zeros(1, 768))
        self.criterion = nn.CrossEntropyLoss()

        if is_freeze_blip:
            self._freeze(self.blip)
            if fine_tune_visual_encoder:
                for n, p in self.blip.visual_encoder.named_parameters():
                    p.requires_grad = True
            if fine_tune_text_encoder:
                for n, p in self.blip.text_encoder.named_parameters():
                    p.requires_grad = True

    def forward(self, text_encodeds, img_encodeds):
        image_embeds = self.blip.visual_encoder(img_encodeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(img_encodeds.device)

        kb_embeds = text_encodeds["kb_embeddings"].to(img_encodeds.device).to(torch.float32)
        cnn_out = self.text_cnn(kb_embeds)  # (total_kb, 768)
        kb_true_annotation_nums = text_encodeds["kb_true_annotation_nums"]
        kb_embeds = []
        # max num of kb entities in each sample is 32
        kb_atts = torch.zeros(image_atts.size(0), 32, dtype=torch.long).to(img_encodeds.device)
        for i in range(len(kb_true_annotation_nums) - 1):
            start, end = kb_true_annotation_nums[i], kb_true_annotation_nums[i + 1]
            if start == end:
                kb_embeds.append(self.null_entities)
            else:
                kb_embeds.append(cnn_out[start:end])
            kb_atts[i, : end - start] = 1
        kb_embeds = pad_sequence(kb_embeds, batch_first=True, padding_value=0)

        image_kb_embeds = torch.cat([image_embeds, kb_embeds], dim=1)
        image_kb_atts = torch.cat([image_atts, kb_atts], dim=1)
        # pad size might be smaller than max length
        image_kb_atts = image_kb_atts[:, : image_kb_embeds.size(1)]

        text_encodeds.input_ids[:, 0] = self.blip.tokenizer.enc_token_id
        output = self.blip.text_encoder(
            text_encodeds.input_ids,
            attention_mask=text_encodeds.attention_mask,
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
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "simple_blip.yaml")
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
