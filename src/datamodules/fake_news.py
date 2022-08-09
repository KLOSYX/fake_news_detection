from pathlib import Path
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoTokenizer


class WeiboDataset:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["title"]
        img_name = self.data.iloc[idx]["imgs"][0]
        img_path = self.img_path / img_name
        img = Image.open(img_path)
        label = self.data.iloc[idx]["label"]
        return text, self.transforms(img).unsqueeze(0), torch.tensor([label], dtype=torch.long)


class Collector:
    def __init__(self, tokenizer: str, processor: str, max_length: int = 200) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir="/data/.cache")
        self.processor = (
            AutoFeatureExtractor.from_pretrained(processor, cache_dir="/data/.cache")
            if processor is not None
            else None
        )
        self.max_length = max_length

    def __call__(self, data: List):
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


class MultiModalData(pl.LightningDataModule):
    def __init__(
        self,
        img_path: str,
        train_path: str,
        test_path: str,
        tokenizer_name: str = "bert-base-chinese",
        processor_name: str = None,
        dataset_name: str = "weibo",
        max_length: int = 200,
        val_set_ratio: int = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.img_path = img_path
        self.train_path = train_path
        self.test_path = test_path
        self.dataset_name = dataset_name
        self.val_set_ratio = val_set_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = Collector(
            tokenizer=tokenizer_name, processor=processor_name, max_length=max_length
        )

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            dataset = (
                WeiboDataset(self.img_path, self.train_path)
                if self.dataset_name == "weibo"
                else None
            )
            total_size = len(dataset)
            val_set_size = int(total_size * self.val_set_ratio)
            print("total_size:", total_size - val_set_size, "val_set_size:", val_set_size)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [total_size - val_set_size, val_set_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = (
                WeiboDataset(self.img_path, self.test_path)
                if self.dataset_name == "weibo"
                else None
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
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
