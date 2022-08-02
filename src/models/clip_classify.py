from argparse import ArgumentParser

from transformers import get_constant_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import confusion_matrix
import nni

from utils.focal_loss import FocalLoss
from models.components.clip_base import ClipBase
from utils.plot_confusion_matrix import plot_confusion_matrix


class ClipClassify(pl.LightningModule):
    def __init__(self,
                 clip_name='openai/clip-vit-base-patch32',
                 bert_name='IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese',
                 num_classes=62,
                 focal_loss=False,
                 learning_rate=1e-5,
                 weight_decay=0.05,
                 dropout_prob=0.1,
                 load_mlm_checkpoint=False,
                 label_smooth=False,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        # model
        self.model = ClipBase(clip_name, bert_name)
        self.projection = nn.Linear(self.model.clip.config.projection_dim * 2, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(256, num_classes)
        self.act = nn.GELU()
        
        self.model.freeze_clip()
        
        # loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if label_smooth else 0) if not focal_loss else FocalLoss(
            alpha=[1] * num_classes, num_classes=num_classes)

        # metric
        self.train_acc = self.get_top_k_metric_collection('Accuracy', prefix='train', num_classes=num_classes, multiclass=True)
        # self.train_f1_score = self.get_top_k_metric_collection('F1Score', prefix='train', num_classes=num_classes, average='macro', multiclass=True)
        self.val_acc = self.get_top_k_metric_collection('Accuracy', prefix='val', num_classes=num_classes, multiclass=True)
        # self.val_f1_score = self.get_top_k_metric_collection('F1Score', prefix='val', num_classes=num_classes, average='macro', multiclass=True)
        self.test_acc = self.get_top_k_metric_collection('Accuracy', prefix='test', num_classes=num_classes, multiclass=True)
        
        
    def get_top_k_metric_collection(self, metric='Accuracy', prefix='train', num_classes=None, top_k=3, **kwargs):
        return torchmetrics.MetricCollection({f'{prefix}/{metric.lower()}_top_{k}': getattr(torchmetrics, metric)(num_classes=num_classes, top_k=k, **kwargs) for k in range(1, top_k + 1)})

    def forward(self, img_encoded, text_encoded):
        img_features, text_features = self.model(text_encoded, img_encoded)
        x = torch.cat([img_features, text_features], dim=1) # [N, hidden_size * 2]
        x = self.dropout(x)
        x = self.act(x)
        x = self.projection(x)
        x = self.dropout(x)
        x = self.act(x) # [N, 256]
        logits = self.classifier(x) # [N, num_classes]
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(
        ), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.num_warmup_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img_encoded, text_encoded, _, labels = batch
        logits = self(img_encoded, text_encoded)
        loss = self.criterion(logits, labels)
        self.log_dict({'train/loss': loss})
        self.log_dict(self.train_acc(logits, labels))
        # self.log_dict(self.train_f1_score(logits, labels))
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img_encoded, text_encoded, _, labels = batch
        logits = self(img_encoded, text_encoded)
        loss = self.criterion(logits, labels)
        self.log_dict({'val/loss': loss})
        self.log_dict(self.val_acc(logits, labels))
        # self.log_dict(self.val_f1_score(logits, labels))
        return (logits, labels)
    
    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if self.hparams.draw_confusion_matrix:
            logits, labels = zip(*outputs)
            file = '/data/clean_raw_text/district_label_map.txt' if 'district' in self.hparams.train_path else '/data/clean_raw_text/street_label_map.txt'
            text_labels = [text.strip('\n') for text in open(file, 'r').readlines()]
            preds = torch.argmax(torch.cat(logits, dim=0), dim=1).detach().cpu().numpy()
            targets = torch.cat(labels, dim=0).detach().cpu().numpy()
            cm = confusion_matrix(targets, preds, labels=range(self.hparams.num_classes), normalize='true')
            if self.hparams.wandb and self.global_rank == 0 and self.hparams.stage == 'fit':
                fig = plot_confusion_matrix(cm, target_names=text_labels, normalize=False)
                self.logger.log_image(key='confusion_matrix', images=[fig])

    def on_validation_epoch_end(self) -> None:
        if self.global_step > 1 and self.hparams.use_nni:
            metric = self.val_acc.compute()['val/accuracy_top_1'].item()
            if self.global_rank == 0:
                nni.report_intermediate_result(metric)
                
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img_encoded, text_encoded, _, labels = batch
        logits = self(img_encoded, text_encoded)
        loss = self.criterion(logits, labels)
        self.log_dict({'test/loss': loss})
        self.log_dict(self.test_acc(logits, labels))
        # self.log_dict(self.val_f1_score(logits, labels))

    @staticmethod
    def add_model_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--draw_confusion_matrix', action='store_true', help='draw confusion matrix or not')
        parser.add_argument('--num_classes', type=int, default=63)
        parser.add_argument('--clip_name', type=str, 
                            default="openai/clip-vit-base-patch32", help='pretrained clip model name')
        parser.add_argument('--bert_name', type=str,
                            default='hfl/chinese-macbert-base', help='text decoder')
        parser.add_argument('--dropout_prob', type=float,
                            default=0.1, help='dropout probability')
        parser.add_argument('--num_warmup_steps', type=int,
                            default=0, help='number of warmup steps')
        parser.add_argument('--label_smooth', action='store_true', help='label smooth')
        parser.add_argument(
            '--focal_loss', action='store_true', help='use focal loss')
        return parser
