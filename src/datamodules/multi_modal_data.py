from pathlib import Path
from typing import Any, List, Optional, Tuple

import jpeg4py as jpeg
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoFeatureExtractor, AutoTokenizer


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
        # image = Image.open(img_path)
        image = jpeg.JPEG(img_path).decode()
        return text, image


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
            else list(images)
        )
        return text_encoded, image_encoded


class MultiModalDatamodule(pl.LightningDataModule):
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
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collector = Collector(
            tokenizer_name=tokenizer_name, processor_name=processor_name, max_length=max_length
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            assert self.train_path is not None, "For stage fit, train path must be provided!"
            dataset = MultiModalDataset(self.train_path)
            val_size = int(len(dataset) * self.val_ratio)
            train_size = len(dataset) - val_size
            print("Train size", train_size, "Val size", val_size)
            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            assert self.test_path is not None, "For stage test, test path must be provided!"
            self.test_data = MultiModalDataset(self.test_path)
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
    dm = MultiModalDatamodule(
        batch_size=4,
        num_workers=0,
        train_path="/data/clean_raw_text/street_labeled_data.json",
        test_path="/data/clean_raw_text/street_labeled_data.json",
        tokenizer_name="IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
    )
    dm.setup()
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)
