from pathlib import Path

import click
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import VaspInput
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm


def prepare_task(structure, relax_path, PSTRESS, NSW):
    mp_set = MPRelaxSet(
        structure,
        user_incar_settings={
            'NSW': NSW,
            'LREAL': False,
            'ISMEAR': 0,
            'EDIFF': 1e-6,
            'EDIFFG': 0.01,
            'PSTRESS': PSTRESS,
        },
        user_potcar_settings={"W": "W_sv"},
        user_potcar_functional="PBE_54",
    )
    vasp = VaspInput(
        incar=mp_set.incar,
        kpoints=mp_set.kpoints,
        poscar=mp_set.poscar,
        potcar=mp_set.potcar,
    )
    vasp.write_input(relax_path)


@click.command
@click.argument("indir")
@click.option("-s", "--nsw", default=0, help="NSW, default 0")
@click.option("-p", "--pstress", default=0, help="PSTRESS (kbar), default 0")
def prepare_batch(indir, nsw: int, pstress: float):
    click.echo(f"You are using {nsw=} {pstress=}")
    indir = Path(indir)
    runtype = ".scf" if nsw <= 1 else ".opt"
    for sf in tqdm(list(indir.glob("*.vasp")) + list(indir.glob("*.cif"))):
        structure = Structure.from_file(sf)
        relax_path = indir.with_suffix(runtype).joinpath(sf.stem)
        relax_path.mkdir(exist_ok=True, parents=True)
        prepare_task(structure, relax_path, pstress, nsw)


if __name__ == '__main__':
    prepare_batch()
