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

    def _get_dataset(self, stage: str = "fit") -> Dataset:
        raise NotImplementedError("get_dataset_args must be implemented!")

    def _init_collector(self) -> None:
        raise NotImplementedError("_init_collector must be implemented!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.collector = self._init_collector()
        if stage == "fit" or stage is None:
            dataset = self._get_dataset(stage)
            val_size = int(len(dataset) * self.val_ratio)
            train_size = len(dataset) - val_size
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
