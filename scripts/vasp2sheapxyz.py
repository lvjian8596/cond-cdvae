# sheap xyz
# Lattice Properties name pressure energy spacegroup(symb) times_found=1
# SOPA-n*-l*-c*-g*             pbc
#         cutoff sigma

import pickle
from itertools import chain
from pathlib import Path

import click
import numpy as np
from ase.io import read, write
from ase.spacegroup import get_spacegroup
from dscribe.descriptors import SOAP
from joblib import Parallel, delayed
from tqdm import tqdm


def save_soap_desc(fdir, scale_volperatom=None, soapkwargs={}):
    fdir = Path(fdir)
    assert (scale_volperatom is None) or (
        isinstance(scale_volperatom, float)
    ), "scalevolume should be none or float"
    soapdesc = SOAP(**soapkwargs)

    soapdict = {}
    for fname in chain(
        fdir.rglob("POSCAR"),
        fdir.rglob("POSCAR_*"),
        fdir.rglob("*.vasp"),
        fdir.rglob("poscar_*"),
        fdir.rglob("contcar_*"),
    ):
        atoms = read(fname, format="vasp")
        if scale_volperatom is not None:
            volume = scale_volperatom * len(atoms)
            cell = atoms.get_cell() * (volume / atoms.get_volume()) ** (1 / 3)
            atoms.set_cell(cell)
        name = str(fname.relative_to(fdir))
        soapdict[name] = soapdesc.create(atoms)
    soap_desc = {"soapkwargs": soapkwargs, "soapdict": soapdict}

    if scale_volperatom is not None:
        desc_savename = f"soap_desc+{scale_volperatom}.pkl"
    else:
        desc_savename = "soap_desc.pkl"
    with open(fdir.joinpath(desc_savename), 'wb') as f:
        pickle.dump(soap_desc, f)

    return soap_desc


def parse_soapdict(cwd, soappkl, soapkeyname, fname, soapdesc: np.ndarray):
    fname = soappkl.parent.joinpath(fname)
    atoms = read(fname)
    spg = get_spacegroup(atoms).symbol.replace(" ", "")
    name = str(fname.relative_to(cwd)).replace("/", "#")
    atoms.info["name"] = f"{name}"
    atoms.info[soapkeyname] = " ".join(map(str, soapdesc))
    atoms.info["energy"] = 0.0
    atoms.info["spacegroup"] = spg
    atoms.info["times_found"] = 1
    return atoms


def soappkl2xyz(njobs, soappkllist):
    cwd = Path().cwd()
    atomslist = []
    for soappkl in soappkllist:
        soappkl = Path(soappkl).resolve()
        with open(soappkl, "rb") as f:
            data = pickle.load(f)
            soapkwargs = data["soapkwargs"]
            soapdict = data["soapdict"]
        soapkeyname = (
            f"SOAP"
            f"-n{soapkwargs.get('n_max')}-"
            f"l{soapkwargs.get('l_max')}"
            f"-c{soapkwargs.get('r_cut')}"
            f"-g{soapkwargs.get('sigma', 1.0)}"
        )

        atomslist += Parallel(njobs, backend="multiprocessing")(
            delayed(parse_soapdict)(cwd, soappkl, soapkeyname, fname, soapdesc)
            for fname, soapdesc in tqdm(
                soapdict.items(),
                total=len(soapdict),
                desc=str(soappkl.relative_to(cwd)),
            )
        )

    fout = "soap.xyz"
    write(fout, atomslist, format="extxyz")


@click.command()
@click.argument('soappkllist', nargs=-1)
@click.option("-j", "--njobs", type=int, default=1)
def main(njobs, soappkllist):
    soappkllist = [Path(soappkl) for soappkl in soappkllist]
    soappkl2xyz(njobs, soappkllist)


if __name__ == "__main__":
    main()
