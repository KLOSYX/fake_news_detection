from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoTokenizer

from src.datamodules.components.dm_base import DatamoduleBase


class WeiboDataset(Dataset):
    def __init__(self, img_path: str, data_path: str) -> None:
        assert isinstance(img_path, str) and isinstance(data_path, str), "path must be a string"
        self.data = pd.read_json(data_path, lines=True)
        self.img_path = Path(img_path)
        # apply image transforms
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomPerspective(p=0.2),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("data size:", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        text = self.data.iloc[idx]["title"]
        img_name = self.data.iloc[idx]["imgs"][0]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        return (
            text,
            self.transforms(img).unsqueeze(0),
            torch.tensor([label], dtype=torch.long),
        )


class TwitterDataset(WeiboDataset):
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        text = self.data.iloc[idx]["text"]
        img_name = self.data.iloc[idx]["imgs"][0]
        img_path = self.img_path / img_name
        img = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        return (
            text,
            self.transforms(img).unsqueeze(0),
            torch.tensor([label], dtype=torch.long),
        )


class Collector:
    def __init__(self, tokenizer: Any, processor: Optional[str], max_length: int = 200) -> None:
        self.tokenizer = tokenizer
        self.processor = (
            AutoFeatureExtractor.from_pretrained(processor, cache_dir="~/.cache")
            if processor is not None
            else None
        )
        self.max_length = max_length

    def __call__(self, data: List) -> Tuple:
        texts, imgs, labels = zip(*data)
        text_encodeds = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        img_encodeds = (
            self.processor(imgs) if self.processor is not None else torch.cat(imgs, dim=0)
        )
        # image agumentation here, todo...
        labels = torch.cat(labels)
        return text_encodeds, img_encodeds, labels


class MultiModalData(DatamoduleBase):
    def __init__(
        self,
        img_path: str,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        val_set_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
        tokenizer_name: str = "bert-base-chinese",
        processor_name: Optional[str] = None,
        max_length: int = 200,
        dataset_name: str = "weibo",
    ):
        super().__init__(val_set_ratio, batch_size, num_workers)
        self.img_path = img_path
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="~/.cache")
        self.processor_name = processor_name
        self.max_length = max_length
        if dataset_name == "weibo":
            self.dataset_cls = WeiboDataset
        elif dataset_name == "twitter":
            self.dataset_cls = TwitterDataset

    def _get_collector(self) -> Any:
        return Collector(
            tokenizer=self.tokenizer,
            processor=self.processor_name,
            max_length=self.max_length,
        )

    def _get_dataset(self, stage: Optional[str] = "fit") -> Dataset:
        return self.dataset_cls(
            img_path=self.img_path,
            data_path=self.train_path if stage == "fit" or stage is None else self.test_path,
        )


# debug
if __name__ == "__main__":
    data = MultiModalData(
        img_path="/data/fake_news/datasets/MM17-WeiboRumorSet/images",
        train_path="/data/fake_news/datasets/MM17-WeiboRumorSet/train_data.json",
        test_path="/data/fake_news/datasets/MM17-WeiboRumorSet/test_data.json",
    )
    data.setup("fit")
    it = iter(data.train_dataloader())
    while True:
        item = next(it)
