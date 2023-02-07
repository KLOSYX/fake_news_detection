from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split


class DatamoduleBase(pl.LightningDataModule):
    def __init__(
        self,
        val_ratio: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        super().__init__()
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data: Dataset | None = None
        self.val_data: Dataset | None = None
        self.test_data: Dataset | None = None

        self.collector = None

    def _get_dataset(self, stage: str | None = "fit") -> Dataset:
        raise NotImplementedError("get_dataset_args must be implemented!")

    def _get_collector(self) -> Any:
        raise NotImplementedError("_init_collector must be implemented!")

    def setup(self, stage: str | None = None) -> None:
        self.collector = self._get_collector()
        if stage == "fit" or stage is None:
            dataset = self._get_dataset(stage)
            val_size = max(int(len(dataset) * self.val_ratio), 1)
            train_size = len(dataset) - val_size
            assert train_size >= 1 and val_size >= 1, "Train size or val size is smaller than 1!"
            print("Train size", train_size, "Val size", val_size)
            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test_data = self._get_dataset(stage)
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
