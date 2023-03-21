import pickle
import sys
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read, write
from eval_utils import get_crystals_list


def save_gen_structure(gen_pt_file):
    gen_data = torch.load(Path(gen_pt_file))
    gen = gen_data
    c_list = get_crystals_list(
        gen['frac_coords'][0], gen['atom_types'][0], gen['lengths'][0], gen['angles'][0], gen['num_atoms'][0]
    )
    a_list = [
        Atoms(c['atom_types'], scaled_positions=c['frac_coords'], cell=c['lengths'].tolist() + c['angles'].tolist())
        for c in c_list
    ]
    # pickle ase Atoms list
    with open(Path(gen_pt_file).with_name('gen.pkl'), 'wb') as f:
        pickle.dump(a_list, f)
    # save to gen/
    save_path = Path(gen_pt_file).with_name('gen')
    save_path.mkdir(exist_ok=True)
    for i, atoms in enumerate(a_list):
        write(save_path.joinpath(f'{i}.vasp'), atoms, direct=True)

    return a_list


if __name__ == '__main__':
    gen_pt_file = Path(sys.argv[1])
    if not gen_pt_file.is_absolute():
        raise OSError('argv[1] should be absolute')
    save_gen_structure(gen_pt_file)
