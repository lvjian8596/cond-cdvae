import unittest
from itertools import chain
from types import SimpleNamespace

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pymatgen.core.composition import Composition

from cdvae.common.utils import PROJECT_ROOT
from cdvae.pl_modules.conditioning import (
    AggregateConditioning,
    AtomwiseConditioning,
    MultiEmbedding,
)


class TestModelCond(unittest.TestCase):
    device = "cuda"

    def setUp(self):
        with hydra.initialize_config_dir(
            version_base="1.1", config_dir=str(PROJECT_ROOT / "conf")
        ):
            self.cfg = hydra.compose(config_name="default", overrides=["data=carbon"])
            self.datamodule: pl.LightningDataModule = hydra.utils.instantiate(
                self.cfg.data.datamodule, _recursive_=False
            )
            self.datamodule.setup()

            self.B = self.cfg.data.datamodule.batch_size.train
            self.cond_dim = self.cfg.model.conditions.cond_dim
            self.latent_dim = self.cfg.model.latent_dim

            self.batch = next(iter(self.datamodule.train_dataloader()))
            self.nnodes = self.batch.batch.shape[0]
            self.conditions = {
                'composition': (self.batch.atom_types, self.batch.num_atoms),
                'energy_per_atom': self.batch.energy_per_atom,
            }
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

    def test_03_multiemb(self):
        multiemb: MultiEmbedding = hydra.utils.instantiate(
            self.cfg.model.conditions, _recursive_=False
        )
        cond_vec = multiemb(self.conditions)
        self.assertEqual(cond_vec.shape, (self.B, self.cond_dim))
        # cuda
        device = 'cuda'
        multiemb = multiemb.to(device)
        conditions = {
            'composition': (
                self.batch.atom_types.to(device),
                self.batch.num_atoms.to(device),
            ),
            'energy_per_atom': self.batch.energy_per_atom.to(device),
        }
        cond_vec = multiemb(conditions)
        self.assertEqual(cond_vec.shape, (self.B, self.cond_dim))

    def test_04_agg(self):
        multiemb: MultiEmbedding = hydra.utils.instantiate(
            self.cfg.model.conditions, _recursive_=False
        )
        cond_vec = multiemb(self.conditions)

        aggcond = AggregateConditioning(self.cond_dim, self.latent_dim)
        agg_latent = aggcond(cond_vec, self.latent)

        self.assertEqual(agg_latent.shape, (self.B, self.latent_dim))

    def test_05_atomwisecond(self):
        multiemb: MultiEmbedding = hydra.utils.instantiate(
            self.cfg.model.conditions, _recursive_=False
        )
        cond_vec = multiemb(self.conditions)

        atomwisecond = AtomwiseConditioning(
            self.cond_dim, self.cfg.model.encoder.hidden_channels
        )
        atomwisecondvec = atomwisecond(
            cond_vec, self.batch.atom_types, self.batch.num_atoms
        )

        self.assertEqual(
            atomwisecondvec.shape,
            (self.nnodes, self.cfg.model.encoder.hidden_channels),
        )

    def test_06_model_and_gen_shape(self):
        model: pl.LightningModule = hydra.utils.instantiate(
            self.cfg.model,
            optim=self.cfg.optim,
            data=self.cfg.data,
            logging=self.cfg.logging,
            _recursive_=False,
        )
        # build_conditions
        conditions = model.build_conditions(self.batch)
        self.assertEqual(conditions, self.conditions)
        # encode
        cond_vec = model.multiemb(conditions)
        mu, log_var, z = model.encode(self.batch, cond_vec)
        self.assertEqual(z.shape, self.latent.shape)
        # agg_cond
        cond_z = model.agg_cond(cond_vec, z)
        self.assertEqual(cond_z.shape, self.latent.shape)
        # decode_stats
        pred_lengths_and_angles, pred_lengths, pred_angles = model.decode_stats(
            cond_z,
            self.batch.num_atoms,
            self.batch.lengths,
            self.batch.angles,
            False,
        )
        self.assertEqual(pred_lengths_and_angles.shape, (self.B, 6))
        # decoder
        pred_cart_coord_diff, _ = model.decoder(
            cond_z,
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
        comp = Composition('H2O')
        each_atom_types = list(
            chain.from_iterable(
                [elem.number] * int(n)
                for elem, n in Composition(
                    comp.get_integer_formula_and_factor()[0]
                ).items()
            )
        )
        num_atoms = torch.tensor(
            [len(each_atom_types)] * n_sample,
            device=model.device,
            dtype=torch.get_default_dtype(),
        )
        atom_types = torch.tensor(
            each_atom_types * n_sample,
            device=model.device,
            dtype=torch.get_default_dtype(),
        )
        sample_conditions = {
            'composition': (atom_types, num_atoms),
            'energy_per_atom': torch.ones(n_sample, 1, device=model.device) * 0.1,
        }

        sample_dict = model.sample(sample_conditions, ld_kwargs)
        self.assertEqual(sample_dict['frac_coords'].shape, (num_atoms, 3))


def main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestModelCond))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
