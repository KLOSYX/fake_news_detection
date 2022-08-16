from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, file_path: str):
        super().__init__()
        file_path = Path(file_path)
        assert file_path.exists(), "Data path not exists!"
        data = pd.read_json(file_path, lines=True)
        assert hasattr(data, "text") and hasattr(data, "label")
        self.data = data
        print("Dataset total size", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Any:
        text, class_id = self.data.iloc[item]["text"], self.data.iloc[item]["label"]
        return text, class_id


class Collector:
    def __init__(
        self,
        tokenizer_name: str = "bert-base-chinese",
        max_length: int = 200,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="/data/.cache")
        self.max_length = max_length

    def __call__(self, data: List[Any]) -> Tuple[Any, torch.Tensor]:
        texts, class_ids = zip(*data)
        text_encoded = self.tokenizer(
            list(texts),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return text_encoded, torch.tensor(class_ids, dtype=torch.long)


class TextClassificationDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        tokenizer_name: str = "bert-base-chinese",
        max_length: int = 200,
        val_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collector = Collector(tokenizer_name=tokenizer_name, max_length=max_length)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            assert self.train_path is not None, "For stage fit, train path must be provided!"
            dataset = TextDataset(self.train_path)
            val_size = int(len(dataset) * self.val_ratio)
            train_size = len(dataset) - val_size
            print("Train size", train_size, "Val size", val_size)
            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            assert self.test_path is not None, "For stage test, test path must be provided!"
            self.test_data = TextDataset(self.test_path)
            print("Test size", len(self.test_data))

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
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collector,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collector,
        )


if __name__ == "__main__":
    # debug
    dm = TextClassificationDatamodule(
        batch_size=4,
        num_workers=0,
        train_path="/data/proj/henyang/dataset/kb_classification_train_data.json",
        test_path="/data/proj/henyang/dataset/kb_classification_train_data.json",
        tokenizer_name="hfl/chinese-roberta-wwm-ext",
    )
    dm.setup()
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)
