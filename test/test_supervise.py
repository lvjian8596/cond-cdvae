import unittest

import hydra
from omegaconf import OmegaConf
from cdvae.common.utils import PROJECT_ROOT


class TestSupervise(unittest.TestCase):
    def setUp(self):
        with hydra.initialize_config_dir(
                version_base='1.1',
                config_dir=str(PROJECT_ROOT / "conf"),
        ):
            self.cfg = hydra.compose(
                config_name="default",
                overrides=["data=carbon"],
            )
        self.model = hydra.utils.instantiate(self.cfg.model, _recursive_=False)

    def tearDown(self):
        pass

    def test_01_supervise(self):
        pass


def main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSupervise))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
