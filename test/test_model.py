import unittest
from types import SimpleNamespace

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from cdvae.common.data_utils import get_scaler_from_data_list
from cdvae.common.utils import PROJECT_ROOT


class TestModel(unittest.TestCase):
    def setUp(self):
        with hydra.initialize_config_dir(version_base='1.1',
                                         config_dir=str(PROJECT_ROOT / "conf")):
            self.cfg = hydra.compose(config_name="default",
                                     overrides=["data=carbon"])

    def tearDown(self):
        pass

    def test_01_dataset(self):
        train_dataset = hydra.utils.instantiate(
            self.cfg.data.datamodule.datasets.train, )
        lattice_scaler = get_scaler_from_data_list(
            train_dataset.cached_data,
            key='scaled_lattice',
        )
        prop_scalers = [
            get_scaler_from_data_list(train_dataset.cached_data, key=p)
            for p in train_dataset.prop
        ]
        train_dataset.lattice_scaler = lattice_scaler
        train_dataset.prop_scalers = prop_scalers
        self.assertEqual(len(train_dataset), 6091)

    def test_02_datamodule(self):
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(
            self.cfg.data.datamodule,
            _recursive_=False,
        )
        datamodule.setup()
        trn_batch = next(iter(datamodule.train_dataloader()))
        self.assertEqual(
            trn_batch.energy_per_atom.shape,
            (self.cfg.data.datamodule.batch_size.train, 1),
        )
        test_batch = next(iter(datamodule.test_dataloader()[0]))
        self.assertEqual(
            test_batch.energy_per_atom.shape,
            (self.cfg.data.datamodule.batch_size.test, 1),
        )

    @unittest.skip('')
    def test_03_model(self):
        model: pl.LightningModule = hydra.utils.instantiate(
            self.cfg.model,
            optim=self.cfg.optim,
            data=self.cfg.data,
            logging=self.cfg.logging,
            _recursive_=False,
        )
        model(self.batch)


def main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
