Clone from [Xie](https://github.com/txie-93/cdvae)

# Crystal Diffusion Variational AutoEncoder

This software implementes Crystal Diffusion Variational AutoEncoder (CDVAE), which generates the periodic structure of materials.

It has several main functionalities:

- Generate novel, stable materials by learning from a dataset containing existing material structures.
- Generate materials by optimizing a specific property in the latent space, i.e. inverse design.

[[Paper]](https://arxiv.org/abs/2110.06197) [[Datasets]](data/)

<p align="center">
  <img src="assets/illustrative.png" /> 
</p>

<p align="center">
  <img src="assets/Tm4Ni4As4.gif" width="200">
</p>


## Table of Contents

- [Crystal Diffusion Variational AutoEncoder](#crystal-diffusion-variational-autoencoder)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Install with pip](#install-with-pip)
    - [Install with conda](#install-with-conda)
    - [Setting up environment variables](#setting-up-environment-variables)
  - [Datasets](#datasets)
  - [Training CDVAE](#training-cdvae)
    - [Training without a property predictor](#training-without-a-property-predictor)
    - [Training with a property predictor](#training-with-a-property-predictor)
  - [Generating materials](#generating-materials)
  - [Evaluating model](#evaluating-model)
  - [Authors and acknowledgements](#authors-and-acknowledgements)
  - [Citation](#citation)
  - [Contact](#contact)

## Installation

### Install with pip

CUDA11.8

It is suggested to use `conda` (by [conda](https://conda.io/docs/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
to create a python3.10 environment first,
then run the following `pip` commands in this environment.

```bash
pip install torch torchaudio torchvision -i https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install lightning torch_geometric
pip install ase black hydra-core matminer matplotlib networkx omegaconf p-tqdm pandas pyarrow
pip install pymatgen python-dotenv scikit-learn scipy smact sympy tqdm wandb yapf
pip install -e .
```

### Install with conda

CUDA11.7

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html).

```bash
conda env create -f environment-v2.yaml
pip install -e .
```

### Setting up environment variables

Modify the following environment variables in file `.env`.

- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WANDB`: path to a folder to store wandb outputs

```env
PROJECT_ROOT="/home/..."
HYDRA_JOBS="/home/..."
WANDB="/home/..."
```

## Datasets

All datasets are directly available on `data/` with train/valication/test splits. You don't need to download them again. If you use these datasets, please consider to cite the original papers from which we curate these datasets.

Find more about these datasets by going to our [Datasets](data/) page.

## Training CDVAE

### Training without a property predictor

To train a CDVAE, run the following command:

```bash
python cdvae/run.py \
    model=vae/vae_nocond \  # vae is default
    project=... group=... expname=... \
    data=... \
    optim.optimizer.lr=1e-7 optim.lr_scheduler.min_lr=1e-7 \
    data.teacher_forcing_max_epoch=250 data.train_max_epochs=4000 \
    model.beta=0.01 \
    model.fc_num_layers=1 model.latent_dim=... 
    model.hidden_dim=... model.lattice_dropout=... \  # MLP part
    model.hidden_dim=... model.latent_dim=... \
    [model.conditions.cond_dim=...] \
```

To train with multi-gpu:
```bash
CUDA_VISIBLE_DEVICES=0,1 python ... \
    train.pl_trainer.devices=2 \
    +train.pl_trainer.strategy=ddp_find_unused_parameters_true
```

To use other datasets, use `data=carbon` and `data=mp_20` instead.
CDVAE uses [hydra](https://hydra.cc) to configure hyperparameters, and users can
modify them with the command line or configure files in `conf/` folder.

After training, model checkpoints can be found in `$HYDRA_JOBS/singlerun/project/group/expname`.

## Generating materials

To generate materials, run recon first:

```bash
python scripts/evaluate.py --model_path MODEL_PATH --tasks recon
```

then

```bash
python scripts/evaluate.py --model_path MODEL_PATH --tasks gen \
    [--formula=H2O/--train_data=*.pkl] \  # if composition condition
    [--energy=-1/--energy_per_atom=-1] \  # if energy condition
    --batch_size=50
```

`MODEL_PATH` will be the path to the trained model. Users can choose one or several of the 3 tasks:

- `recon`: reconstruction, reconstructs all materials in the test data. Outputs can be found in `eval_recon.pt`l
- `gen`: generate new material structures by sampling from the latent space. Outputs can be found in `eval_gen.pt`.
- `opt`: generate new material strucutre by minimizing the trained property in the latent space (requires `model.predict_property=True`). Outputs can be found in `eval_opt.pt`.

`eval_recon.pt`, `eval_gen.pt`, `eval_opt.pt` are pytorch pickles files containing multiple tensors that describes the structures of `M` materials batched together. Each material can have different number of atoms, and we assume there are in total `N` atoms. `num_evals` denote the number of Langevin dynamics we perform for each material.

- `frac_coords`: fractional coordinates of each atom, shape `(num_evals, N, 3)`
- `atom_types`: atomic number of each atom, shape `(num_evals, N)`
- `lengths`: the lengths of the lattice, shape `(num_evals, M, 3)`
- `angles`: the angles of the lattice, shape `(num_evals, M, 3)`
- `num_atoms`: the number of atoms in each material, shape `(num_evals, M)`

## Evaluating model

To compute evaluation metrics, run the following command:

```
python scripts/compute_metrics.py --root_path MODEL_PATH --tasks recon gen opt
```

`MODEL_PATH` will be the path to the trained model. All evaluation metrics will be saved in `eval_metrics.json`.

## Authors and acknowledgements

The software is primary written by [Tian Xie](www.txie.me), with signficant contributions from [Xiang Fu](https://xiangfu.co/).

The GNN codebase and many utility functions are adapted from the [ocp-models](https://github.com/Open-Catalyst-Project/ocp) by the [Open Catalyst Project](https://opencatalystproject.org/). Especially, the GNN implementations of [DimeNet++](https://arxiv.org/abs/2011.14115) and [GemNet](https://arxiv.org/abs/2106.08903) are used.

The main structure of the codebase is built from [NN Template](https://github.com/lucmos/nn-template).

For the datasets, [Perov-5](data/perov_5) is curated from [Perovksite water-splitting](https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html), [Carbon-24](data/carbon_24) is curated from [AIRSS data for carbon at 10GPa](https://archive.materialscloud.org/record/2020.0026/v1), [MP-20](data/mp_20) is curated from [Materials Project](https://materialsproject.org).

## Citation

Please consider citing the following paper if you find our code & data useful.

```
@article{xie2021crystal,
  title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
  author={Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2110.06197},
  year={2021}
}
```

## Contact

Please leave an issue or reach out to Tian Xie (txie AT csail DOT mit DOT edu) if you have any questions.
