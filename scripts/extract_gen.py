import pickle
from pathlib import Path

import click
import torch
from ase import Atoms
from ase.io import write


@click.command(
    help="Extract `eval_gen_*.pt` to `eval_gen*/eval_gen_*.pt.pkl`"
    "and `eval_gen*/gen`. If pkl with same name exists, no extraction will perform. "
    "File name of gen/*.vasp will auto-add."
)
@click.argument("gen_pt_files", nargs=-1)
def save_gen_structure(gen_pt_files):
    gen_pt_files = [Path(f).resolve() for f in gen_pt_files]

    from eval_utils import get_crystals_list  # change working dir

    for gen_pt_file in gen_pt_files:
        click.echo(f"{gen_pt_file}")
        gen = torch.load(gen_pt_file)
        extract_dir = gen_pt_file.with_name(gen_pt_file.stem)
        extract_dir.mkdir(exist_ok=True)
        pkl = extract_dir.joinpath(gen_pt_file.name + ".pkl")
        # Only run if no corresponding pkl file exists
        if not pkl.exists():
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
            # pickle ase Atoms list
            with open(pkl, 'wb') as f:
                pickle.dump(a_list, f)
            # save to gen/
            save_path = extract_dir.joinpath('gen')
            save_path.mkdir(exist_ok=True)
            vasplist = list(save_path.glob("*.vasp"))
            max_stem = max([-1] + [int(f.stem) for f in vasplist])
            for i, atoms in enumerate(a_list, max_stem + 1):
                write(save_path.joinpath(f'{i}.vasp'), atoms, direct=True)


if __name__ == '__main__':
    save_gen_structure()
