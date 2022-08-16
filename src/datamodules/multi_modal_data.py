from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer

from src.datamodules.dm_base import DatamoduleBase


class MultiModalDataset(Dataset):
    def __init__(self, file_path: str):
        super().__init__()
        file_path = Path(file_path)
        assert file_path.exists(), "Data path not exists!"
        data = pd.read_json(file_path, lines=True)
        data = data[data.img.apply(lambda x: x is not None and len(x) > 0)]
        data = data[["text", "img"]]
        self.data = data
        print("Dataset total size", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Any:
        text, img_path = self.data.iloc[item]["text"], self.data.iloc[item]["img"]
        raw_image = Image.open(img_path)
        return text, raw_image


class Collector:
    def __init__(
        self,
        tokenizer_name: str = "bert-base-chinese",
        processor_name: Optional[str] = None,
        max_length: int = 200,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="/data/.cache")
        self.processor = (
            AutoFeatureExtractor.from_pretrained(processor_name, cache_dir="/data/.cache")
            if processor_name is not None
            else None
        )
        self.max_length = max_length

    def __call__(self, data: List[Any]) -> Tuple[Any, Any]:
        texts, images = zip(*data)
        text_encoded = self.tokenizer(
            list(texts),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        image_encoded = (
            self.processor(images, return_tensors="pt")
            if self.processor is not None
            else torch.cat([img.unsqueeze(0) for img in images], dim=0)
        )
        return text_encoded, image_encoded


class MultiModalDatamodule(DatamoduleBase):
    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        tokenizer_name: str = "bert-base-chinese",
        processor_name: Optional[str] = None,
        max_length: int = 200,
        val_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        super().__init__(
            MultiModalDataset,
            train_path=train_path,
            test_path=test_path,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer_name = tokenizer_name
        self.processor_name = processor_name
        self.max_length = max_length

    def _get_dataset_args(self, stage: str = "fit") -> Dict:
        return dict(
            file_path=self.train_path if stage == "fit" else self.test_path,
        )

    def _init_collector(self) -> None:
        self.collector = Collector(
            tokenizer_name=self.tokenizer_name,
            processor_name=self.processor_name,
            max_length=self.max_length,
        )


if __name__ == "__main__":
    # debug
    dm = MultiModalDatamodule(
        batch_size=4,
        num_workers=0,
        train_path="/data/clean_raw_text/street_labeled_data.json",
        test_path="/data/clean_raw_text/street_labeled_data.json",
        tokenizer_name="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
        processor_name="openai/clip-vit-base-patch32",
    )
    dm.setup()
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)
