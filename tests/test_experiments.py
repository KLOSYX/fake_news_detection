from pathlib import Path
from typing import List, Tuple

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from src.train import train


def get_train_experiment_cfgs() -> List[Tuple[str, DictConfig]]:
    config_path = Path(root / "configs" / "experiment")
    cfgs = []
    for config in config_path.glob("**/*.yaml"):
        if "example.yaml" in str(config):  # do not test example.yaml
            continue
        with initialize(
            version_base="1.2", config_path="../configs", job_name="test_train_experiment"
        ):
            experiment = str(config).split("experiment/")[-1]
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[f"experiment={experiment}"],
            )
            sample_path = cfg.datamodule.get("_target_").split(".")[-2]
            config_path = root / "tests" / "datamodule" / sample_path / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    override_cfg = OmegaConf.load(f)
                cfg.datamodule.merge_with(override_cfg)
            with open_dict(cfg):
                cfg.paths.root_dir = str(root)
                cfg.trainer.fast_dev_run = True
                cfg.trainer.accelerator = "auto"
                cfg.trainer.strategy = None
                cfg.trainer.devices = 1
                cfg.extras.print_config = False
                cfg.extras.enforce_tags = False
                cfg.datamodule.num_workers = 0
                cfg.logger = None
            cfgs.append((experiment, cfg))
    return cfgs


@pytest.mark.slow
@pytest.mark.parametrize("experiment,cfg", get_train_experiment_cfgs())
def test_train_experiments(tmp_path, experiment: str, cfg: DictConfig):
    cfg = cfg.copy()
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.paths.log_dir = tmp_path
        cfg.paths.output_dir = tmp_path
    train(cfg)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
