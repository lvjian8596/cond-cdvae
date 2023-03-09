import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from cdvae.common.utils import PROJECT_ROOT
from cdvae.pl_modules.conditioning import (
    AggregateConditioning,
    AtomwiseConditioning,
    MultiEmbedding,
)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)

    B = batch.num_atoms.shape[0]
    nnodes = batch.batch.shape[0]
    cond_dim = cfg.model.conditions.cond_dim

    conditions = {
        'composition': (batch.atom_types, batch.num_atoms),
        'energy_per_atom': batch.energy_per_atom,
    }

    multiemb: MultiEmbedding = hydra.utils.instantiate(
        cfg.model.conditions, _recursive_=False
    )
    cond = multiemb(conditions)  # z(B, cond_dim)
    print(cond.shape)
    assert cond.shape == (B, cond_dim)

    z_dim = 100
    z = torch.randn(B, z_dim)
    aggcond = AggregateConditioning(cfg.model.conditions.cond_dim, z_dim)
    e = aggcond(cond, z)
    print(e.shape)
    assert e.shape == (B, z_dim)

    atomwisecond = AtomwiseConditioning(
        cfg.model.conditions.cond_dim, cfg.model.encoder.hidden_channels
    )
    e = atomwisecond(cond, batch.atom_types, batch.num_atoms)
    print(e.shape)


if __name__ == '__main__':
    main()
