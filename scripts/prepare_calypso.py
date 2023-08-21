import warnings
import shutil
from itertools import product
from pathlib import Path

import click
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Potcar
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm


def potcar2distmat(potcar: Potcar):
    rc = [p.RCORE for p in potcar]
    distmat = np.asarray([(i + j) * 0.529177 for i, j in product(rc, repeat=2)])
    distmat = distmat.reshape(len(rc), len(rc))
    return distmat


def vasp2inputdat(poscar, potcar, distratio=1.0):
    distmat = potcar2distmat(potcar) * distratio
    ds = ""
    for line in distmat:
        ds += " ".join(map(str, line)) + "\n"

    inputdat = (
        f"SystemName = {''.join(poscar.site_symbols)}\n"
        f"NumberOfSpecies = {len(poscar.site_symbols)}\n"
        f"NameOfAtoms = {' '.join(poscar.site_symbols)}\n"
        f"NumberOfAtoms = {' '.join(map(str, poscar.natoms))}\n"
        "NumberOfFormula = 1 1\n"
        f"Volume = {poscar.structure.volume}\n"
        "@DistanceOfIon\n"
        f"{ds}"
        "@End\n"
        "Ialgo = 2\n"
        "PsoRatio = 0.6\n"
        "PopSize = 10\n"
        "ICode = 1\n"
        "NumberOfLbest = 4\n"
        "NumberOfLocalOptim = 3\n"
        "Command = sh submit.sh\n"
        "MaxStep = 5\n"
        "PickUp = F\n"
        "PickStep = 5\n"
        "Parallel = F\n"
        "Split = T\n"
    )
    return inputdat


@click.command
@click.argument("fdir")
@click.option("-r", "--distratio", type=float, help="distance ratio on DistanceOfIon")
def prepare_calypso(fdir, distratio=1.0):
    fdir = Path(fdir)
    for subdir in tqdm(list(fdir.glob("*"))):
        if subdir.is_dir():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                poscar = Poscar.from_file(subdir.joinpath("POSCAR"))
                potcar = Potcar.from_file(subdir.joinpath("POTCAR"))

            calydir = subdir.joinpath("caly")
            calydir.mkdir(exist_ok=True)

            with open(calydir.joinpath("input.dat"), 'w') as f:
                f.write(vasp2inputdat(poscar, potcar, distratio))
            shutil.copy(subdir.joinpath("POTCAR"), calydir)
            shutil.copy(subdir.joinpath("INCAR"), calydir)
            shutil.copy(subdir.joinpath("KPOINTS"), calydir)


if __name__ == "__main__":
    prepare_calypso()
