from pathlib import Path

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cdvae.common.utils import PROJECT_ROOT, log_hyperparameters


def run(cfg: DictConfig):
    print(cfg)
    pass


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()