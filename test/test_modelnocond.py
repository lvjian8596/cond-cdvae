import unittest
from itertools import chain
from types import SimpleNamespace

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pymatgen.core.composition import Composition

from cdvae.common.data_utils import get_scaler_from_data_list
from cdvae.common.utils import PROJECT_ROOT


class TestModelNoCond(unittest.TestCase):
    device = "cuda"

    def setUp(self):
        with hydra.initialize_config_dir(
            version_base='1.1', config_dir=str(PROJECT_ROOT / "conf")
        ):
            self.cfg = hydra.compose(
                config_name="default", overrides=["data=carbon", "model=vae_nocond"]
            )
            self.datamodule = pl.LightningDataModule = hydra.utils.instantiate(
                self.cfg.data.datamodule, _recursive_=False
            )
            self.datamodule.setup()

            self.B = self.cfg.data.datamodule.batch_size.train
            self.latent_dim = self.cfg.model.latent_dim

            self.batch = next(iter(self.datamodule.train_dataloader()))
            self.nnodes = self.batch.batch.shape[0]
            self.latent = torch.randn(self.B, self.latent_dim)

    def tearDown(self):
        pass

    def test_01_load_data(self):
        self.assertEqual(self.batch.num_atoms.shape[0], self.B)

    def test_02_datamodule(self):
        trn_batch = next(iter(self.datamodule.train_dataloader()))
        self.assertEqual(
            trn_batch.energy_per_atom.shape,
            (self.cfg.data.datamodule.batch_size.train, 1),
        )
        test_batch = next(iter(self.datamodule.test_dataloader()[0]))
        self.assertEqual(
            test_batch.energy_per_atom.shape,
            (self.cfg.data.datamodule.batch_size.test, 1),
        )

    def test_03_model(self):
        model: pl.LightningModule = hydra.utils.instantiate(
            self.cfg.model,
            optim=self.cfg.optim,
            data=self.cfg.data,
            logging=self.cfg.logging,
            _recursive_=False,
        )
        # encode
        mu, log_var, z = model.encode(self.batch)
        self.assertEqual(z.shape, self.latent.shape)
        # decode_stats
        (
            num_atoms,
            pred_lengths_and_angles,
            pred_lengths,
            pred_angles,
            composition_per_atom,
        ) = model.decode_stats(
            z,
            self.batch.num_atoms,
            self.batch.lengths,
            self.batch.angles,
            False,
        )
        self.assertEqual(pred_lengths_and_angles.shape, (self.B, 6))
        # decoder
        pred_cart_coord_diff, _ = model.decoder(
            z,
            self.batch.frac_coords,
            self.batch.atom_types,
            self.batch.num_atoms,
            pred_lengths,
            pred_angles,
        )
        self.assertEqual(pred_cart_coord_diff.shape, (self.nnodes, 3))
        # sample
        model = model.to(self.device)
        model.eval()
        ld_kwargs = SimpleNamespace(
            n_step_each=10,
            step_lr=1e-4,
            min_sigma=0,
            save_traj=False,
            disable_bar=False,
        )
        n_sample = 1
        sample_dict = model.sample(n_sample, ld_kwargs)
        self.assertEqual(sample_dict["num_atoms"].shape[0], n_sample)


def main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestModelNoCond))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
