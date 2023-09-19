# NOT finished

from pathlib import Path

import click
import hydra
import torch
import numpy as np
from hydra import compose, initialize_config_dir


def load_model_modified(model_path, data_path):
    from cdvae.common.utils import set_precision

    with initialize_config_dir(str(model_path), version_base="1.1"):
        cfg = compose(config_name='hparams')
        set_precision(cfg.model.get('prec', 32))

        cfg.data.datamodule.datasets.train.path = data_path
        datamodule = hydra.utils.instantiate(
            cfg.data.datamodule, _recursive_=False, scaler_path=model_path
        )
        loader = datamodule.train_dataloader()

        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )

        # dummybatch = next(iter(test_loader))
        # model.forward(dummybatch)  # initialize LazyModel

        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts]
            )
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model = model.__class__.load_from_checkpoint(ckpt, **cfg.model)
        model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
        model.prop_scalers = torch.load(model_path / 'prop_scalers.pt')

    # model = torch.compile(model, mode="reduce-overhead")
    return model, loader, cfg


@click.command
@click.option("-m", "--model_path")
@click.option("-d", "--data_path", help="data table path, eg. *.feather")
def main(model_path, data_path):
    model_path = Path(model_path).resolve()
    print(model_path)
    data_path = str(Path(data_path).resolve())
    model, dataloader, _ = load_model_modified(model_path, data_path)
    print(len(dataloader))

    from evaluate import reconstruction


if __name__ == "__main__":
    main()
