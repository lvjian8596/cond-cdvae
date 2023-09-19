# extract recon pt
# NOT finished

import pickle
from pathlib import Path

import click
import torch
from ase import Atoms
from ase.io import write
from tqdm import tqdm


@click.command
@click.argument("recon_pt_file")
def save_recon_structure(recon_pt_file):
    recon_pt_file = Path(recon_pt_file).resolve()
    recon = torch.load(recon_pt_file)
    print(recon.keys())

    from eval_utils import get_crystals_list  # change working dir

    batch = recon['input_data_batch']
    print(batch)
    true_crystal_array_list = get_crystals_list(
        batch.frac_coords, batch.atom_types, batch.lengths,
        batch.angles, batch.num_atoms)
    print(true_crystal_array_list[0])
    # c_list = get_crystals_list(
    #     recon['frac_coords'][0],
    #     recon['atom_types'][0],
    #     recon['lengths'][0],
    #     recon['angles'][0],
    #     recon['num_atoms'][0],
    # )
    # a_list = [
    #     Atoms(
    #         c['atom_types'],
    #         scaled_positions=c['frac_coords'],
    #         cell=c['lengths'].tolist() + c['angles'].tolist(),
    #     )
    #     for c in tqdm(c_list)
    # ]
    # print(a_list[0])


if __name__ == "__main__":
    save_recon_structure()