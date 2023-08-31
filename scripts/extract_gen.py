import pickle
from pathlib import Path

import click
import torch
from ase import Atoms
from ase.io import write


@click.command
@click.argument("gen_pt_file")
def save_gen_structure(gen_pt_file):
    gen_pt_file = Path(gen_pt_file).resolve()
    gen = torch.load(gen_pt_file)

    from eval_utils import get_crystals_list  # change working dir

    c_list = get_crystals_list(
        gen['frac_coords'][0],
        gen['atom_types'][0],
        gen['lengths'][0],
        gen['angles'][0],
        gen['num_atoms'][0],
    )
    a_list = [
        Atoms(
            c['atom_types'],
            scaled_positions=c['frac_coords'],
            cell=c['lengths'].tolist() + c['angles'].tolist(),
        )
        for c in c_list
    ]
    extract_dir = gen_pt_file.with_name(gen_pt_file.stem)
    extract_dir.mkdir(exist_ok=True)
    # pickle ase Atoms list
    if extract_dir.joinpath('gen.pkl').exists():
        with open(extract_dir.joinpath('gen.pkl'), 'rb') as f:
            privious_list = pickle.load(f)
        a_list = privious_list + a_list
    with open(extract_dir.joinpath('gen.pkl'), 'wb') as f:
        pickle.dump(a_list, f)
    # save to gen/
    save_path = extract_dir.joinpath('gen')
    save_path.mkdir(exist_ok=True)
    vasplist = list(save_path.glob("*.vasp"))
    max_stem = max([-1] + [int(f.stem) for f in vasplist])
    for i, atoms in enumerate(a_list, max_stem + 1):
        write(save_path.joinpath(f'{i}.vasp'), atoms, direct=True)
    return a_list


if __name__ == '__main__':
    save_gen_structure()
