import hydra
import omegaconf

import pytorch_lightning as pl
from cdvae.common.utils import PROJECT_ROOT
from cdvae.pl_modules.conditioning import MultiEmbedding


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)
    # print(cfg.data.conditions)
    # print(cfg.model.conditions)
    # print('composition' in cfg.model.conditions.types)

    multiemb = hydra.utils.instantiate(cfg.model.conditions, _recursive_=False)
    emb = multiemb(batch)
    print(emb.shape)

if __name__ == '__main__':
    main()
