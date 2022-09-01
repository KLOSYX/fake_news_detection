from pathlib import Path
from typing import Any, List, Optional

import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.datamodules.components.dm_base import DatamoduleBase


class SimSenDataset(Dataset):
    def __init__(self, train_path: str) -> None:
        super().__init__()
        train_path = Path(train_path)
        assert train_path.exists(), "train_path must be a valid path!"
        with open(train_path) as f:
            self.data = f.readlines()
        print("Train size", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> List[str]:
        return self.data[index].split("\t")


class Collector:
    def __init__(self, tokenizer_name: str, max_length: int = 128) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="/data/.cache")
        self.max_length = max_length

    def __call__(self, data: List[List[str]]) -> Any:
        pairs = []
        for pair in data:
            pairs.append((pair[0], pair[1]))
            pairs.append((pair[1], pair[0]))
        encoded = self.tokenizer(
            pairs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        input_ids = encoded.input_ids
        attention_mask = self.get_simbert_mask(input_ids, sep_token_id=self.tokenizer.sep_token_id)
        labels = self.get_simbert_labels(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def get_simbert_mask(input_ids: torch.Tensor, sep_token_id: int) -> torch.Tensor:
        sequence_length = input_ids.size(-1)
        attention_masks = []
        for sequence_ids in input_ids:
            first_sep = (sequence_ids == sep_token_id).nonzero()[0][-1].item()
            last_sep = (sequence_ids == sep_token_id).nonzero()[-1][-1].item()
            attention_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.long)
            attention_mask[: last_sep + 1, : first_sep + 1] = 1
            attention_mask[
                first_sep + 1 : last_sep + 1, first_sep + 1 : last_sep + 1
            ] = torch.tril(torch.ones(last_sep - first_sep, last_sep - first_sep), diagonal=0)
            attention_masks.append(attention_mask.unsqueeze(0))
        return torch.cat(attention_masks, dim=0)

    @staticmethod
    def get_simbert_labels(
        input_ids: torch.Tensor, pad_token_id: torch.Tensor, sep_token_id: int = 102
    ) -> torch.Tensor:
        labels = input_ids.clone()
        for label in labels:
            first_sep = (label == sep_token_id).nonzero()[0][0].item()
            label[: first_sep + 1] = -100
        labels[labels == pad_token_id] = -100
        return labels


class SimBertData(DatamoduleBase):
    def __init__(
        self,
        val_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
        train_path: str = "/data/proj/henyang/sim_sentence_data.txt",
        test_path: Optional[str] = None,
        tokenizer_name: str = "hfl/chinese-roberta-wwm-ext",
        max_length: int = 128,
    ):
        super().__init__(val_ratio, batch_size, num_workers)
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

    def _get_dataset(self, stage: Optional[str] = "fit") -> Dataset:
        return (
            SimSenDataset(self.train_path)
            if stage == "fit" or stage is None
            else SimSenDataset(self.test_path)
        )

    def _get_collector(self) -> Any:
        return Collector(self.tokenizer_name, self.max_length)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collector,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collector,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collector,
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "sim_bert_data.yaml")
    dm = hydra.utils.instantiate(cfg)
    dm.setup("fit")
    data = iter(dm.train_dataloader())
    while True:
        item = next(data)
