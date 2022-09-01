from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.datamodules.components.dm_base import DatamoduleBase


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

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        text, class_id = self.data.iloc[idx]["text"], self.data.iloc[idx]["label"]
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


class TextClassificationDatamodule(DatamoduleBase):
    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        val_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
        tokenizer_name: str = "bert-base-chinese",
        max_length: int = 200,
    ):
        super().__init__(
            val_ratio,
            batch_size,
            num_workers,
        )
        self.train_path = train_path
        self.test_path = test_path

        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

    def _get_dataset(self, stage: Optional[str] = "fit") -> Dataset:
        return TextDataset(
            file_path=self.train_path if stage == "fit" or stage is None else self.test_path,
        )

    def _get_collector(self) -> Any:
        return Collector(self.tokenizer_name, self.max_length)


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
