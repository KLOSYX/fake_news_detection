from pathlib import Path

import pytest
import torch
import yaml
from transformers import BatchEncoding, BatchFeature

from src.datamodules.multi_modal_data import MultiModalDatamodule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_fake_news_datamodule(batch_size):
    # check data path
    config_path = Path("configs/datamodule/multi_modal_data.yaml")
    assert config_path.exists()
    with open(config_path) as infile:
        config = yaml.safe_load(infile)
    train_path = Path(config["train_path"])
    test_path = Path(config["train_path"])
    assert train_path.exists() and test_path.exists()

    # init data module
    dm = MultiModalDatamodule(
        train_path=str(train_path),
        test_path=str(test_path),
        batch_size=batch_size,
        tokenizer_name=config["tokenizer_name"],
        processor_name=config["processor_name"],
    )

    # check data loader
    dm.setup()
    assert dm.train_data and dm.val_data and dm.test_data
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # check data type
    batch = next(iter(dm.train_dataloader()))
    text_encodeds, img_encodeds = batch
    assert len(text_encodeds.input_ids) == batch_size
    assert len(img_encodeds) == batch_size or len(img_encodeds.pixel_values) == batch_size
    assert isinstance(text_encodeds, BatchEncoding)
    assert isinstance(img_encodeds, BatchFeature) or img_encodeds[0].dtype == torch.float32
