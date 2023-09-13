from pathlib import Path

import click
from joblib import Parallel, delayed
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import VaspInput
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm


def prepare_task(structure, relax_path, PSTRESS, NSW):
    user_incar_settings = {
        'NSW': NSW,
        'LREAL': False,
        'ISMEAR': 0,
        'EDIFF': 1e-6,
        'EDIFFG': -0.01,
        'PSTRESS': PSTRESS,
        'NCORE': 4,
    }
    if NSW > 1:
        user_incar_settings["ISYM"] = 0

    mp_set = MPRelaxSet(
        structure,
        user_incar_settings=user_incar_settings,
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


def wrapped_prepare_task(indir, sf, nsw, pstress):
    runtype = ".scf" if nsw <= 1 else ".opt"
    structure = Structure.from_file(sf)
    relax_path = indir.with_suffix(runtype).joinpath(sf.stem)
    relax_path.mkdir(exist_ok=True, parents=True)
    prepare_task(structure, relax_path, pstress, nsw)


@click.command
@click.argument("indir")
@click.option("-s", "--nsw", default=0, help="NSW, default 0")
@click.option("-p", "--pstress", default=0, help="PSTRESS (kbar), default 0")
@click.option("-j", "--njobs", default=-1, type=int)
def prepare_batch(indir, nsw: int, pstress: float, njobs: int):
    click.echo(f"You are using {nsw=} {pstress=}")
    indir = Path(indir)
    Parallel(njobs, backend="multiprocessing")(
        delayed(wrapped_prepare_task)(indir, sf, nsw, pstress)
        for sf in tqdm(list(indir.glob("*.vasp")) + list(indir.glob("*.cif")))
    )


if __name__ == '__main__':
    prepare_batch()
