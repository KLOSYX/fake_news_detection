"""Implementation of the EANN model.

original paper: https://dl.acm.org/doi/10.1145/3219819.3219903 original code:
https://github.com/xiaolan98/BDANN-IJCNN2020
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from torch.autograd import Function, Variable
from torchvision.models import VGG19_BN_Weights, vgg19_bn
from transformers import BertConfig, BertModel

from src.models.components.fake_news_base import FakeNewsBase


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -1


def grad_reverse(x):
    return ReverseLayerF.apply(x)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(
        self,
        event_num: int = 40,
        class_num: int = 2,
        hidden_dim: int = 32,
        bert_name: str = "bert-base-uncased",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.event_num = event_num

        self.class_num = class_num
        self.hidden_size = hidden_dim
        self.social_size = 19

        # bert
        bert_model = BertModel.from_pretrained(bert_name, cache_dir=Path.home() / ".cache")
        bert_config = BertConfig.from_pretrained(bert_name, cache_dir=Path.home() / ".cache")
        self.bert_hidden_size = bert_config.hidden_size
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model

        self.dropout = nn.Dropout(dropout)

        # IMAGE
        vgg_19 = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules["6"].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc1 = nn.Linear(num_ftrs,  512)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module("c_fc1", nn.Linear(2 * self.hidden_size, self.class_num))
        # self.class_classifier.add_module("c_softmax", nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module(
            "d_fc1", nn.Linear(2 * self.hidden_size, self.hidden_size)
        )
        self.domain_classifier.add_module("d_relu1", nn.LeakyReLU(True))
        self.domain_classifier.add_module("d_fc2", nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module("d_softmax", nn.Softmax(dim=1))

    @staticmethod
    def mean_average(attention_mask: torch.Tensor, sequence_out: torch.Tensor) -> torch.Tensor:
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
        return bert_out

    def forward(self, text_encodeds, image_encodeds):
        # IMAGE
        image = self.vgg(image_encodeds)  # [N, 512]
        # image = self.image_fc1(image)
        image = F.relu(self.image_fc1(image))

        # last_hidden_state = torch.mean(self.bertModel(**text_encodeds)[0], dim=1, keepdim=False)
        last_hidden_state = self.mean_average(
            text_encodeds.attention_mask, self.bertModel(**text_encodeds)[0]
        )
        text = F.relu(self.fc2(last_hidden_state))
        text_image = torch.cat((text, image), 1)

        # Fake or real
        class_output = self.class_classifier(text_image)
        # Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


class BDANN(FakeNewsBase):
    def __init__(
        self,
        event_num: int = 40,
        class_num: int = 1,
        hidden_dim: int = 512,
        bert_name: str = "bert-base-uncased",
        dropout: float = 0.5,
        lr: float = 0.001,
    ):
        super().__init__()
        self.model = CNN_Fusion(
            event_num,
            class_num,
            hidden_dim,
            bert_name,
            dropout,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, text_encodeds, img_encodeds):
        class_output, domain_output = self.model(text_encodeds, img_encodeds)
        return class_output, domain_output

    def forward_loss(self, batch):
        text_encodeds, img_encodeds, labels, event_labels = batch
        class_output, domain_output = self.forward(text_encodeds, img_encodeds)
        class_output = class_output.squeeze()
        class_loss = self.criterion(class_output, labels.to(torch.float))
        domain_loss = self.domain_criterion(domain_output, event_labels)
        loss = class_loss - domain_loss
        return torch.sigmoid(class_output), labels, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, list(self.model.parameters())), lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1 / (1.0 + 10 * float(epoch) / 100) ** 0.75
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "bdann.yaml")
    dm_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "twitter_event.yaml")
    model = hydra.utils.instantiate(model_cfg)
    dm = hydra.utils.instantiate(dm_cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    model.forward_loss(batch)
    pass
