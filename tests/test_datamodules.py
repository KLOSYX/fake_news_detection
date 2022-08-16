from pathlib import Path
from typing import Dict

import hydra
import pyrootutils
import pytest
import yaml
from omegaconf import OmegaConf

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@pytest.mark.slow
def test_fake_news_datamodule():
    # check data path
    config_path = Path(root / "configs" / "datamodule")
    for config in config_path.glob("*.yaml"):
        config_dict = yaml.safe_load(open(config))
        for k, v in config_dict.items():
            if isinstance(v, str):
                config_dict[k] = v.replace(r"${paths.data_dir}", str(root / "data"))
        cfg = OmegaConf.create(config_dict)
        # init data module
        dm = hydra.utils.instantiate(cfg)

        # check data loader
        if hasattr(dm, "prepare_data"):
            dm.prepare_data()
        dm.setup()
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
