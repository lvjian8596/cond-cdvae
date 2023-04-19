import os

nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import click
import pandas as pd
import quippy
from ase.constraints import UnitCellFilter
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.spacegroup import get_spacegroup
from ase.units import GPa
from joblib import Parallel, delayed


def atoms2cifstring(atoms):
    with io.BytesIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='cif')
        cif = buffer.getvalue().decode()  # byte to string
    return cif


def get_calc_from_env():
    carbon_gap_20_pot = os.getenv("Carbon_GAP_20_pot")
    pot_param_fname = Path(carbon_gap_20_pot).joinpath("Carbon_GAP_20.xml")
    if not pot_param_fname.exists():
        raise Exception("Carbon GAP potential file not exists")
    pot_param_fname = str(pot_param_fname)
    calc = quippy.potential.Potential(param_filename=pot_param_fname)
    return calc


def opt_one(f, rundir, calc, pressure, fmax):
    f = Path(f)
    fname = f.name
    traj = rundir.joinpath(fname).with_suffix(".traj").__str__()
    outp = rundir.joinpath(fname).with_suffix(".xyz")
    p = pressure * GPa

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        atoms = read(f)
    atoms.info['material_id'] = f.stem
    atoms.calc = calc

    with Trajectory(traj, "w", atoms) as trajf:
        ucf = UnitCellFilter(atoms, scalar_pressure=p)
        opt = LBFGS(ucf, logfile=None)
        opt.attach(trajf, interval=10)
        opt.run(fmax=fmax)

    write(outp, atoms)
    cif = atoms2cifstring(atoms)

    ser = pd.Series(
        {
            "material_id": f.stem,
            "nsites": len(atoms),
            "formula": atoms.get_chemical_formula("metal"),
            "spgno": get_spacegroup(atoms).no,
            "energy": atoms.get_potential_energy(),
            "energy_per_atom": atoms.get_potential_energy() / len(atoms),
            "pressure": pressure,
            "volume": atoms.get_volume(),
            "cif": cif,
        }
    )
    return ser


@click.command()
@click.argument("files", nargs=-1)
@click.option("-j", "--jobs", type=int, default=30)
@click.option("-r", "--rundir", default="runquip", help="running dir, default rungulp")
@click.option("-p", "--pressure", type=float, default=0.0, help="pressure in GPa")
@click.option("--fmax", type=float, default=0.01, help="convergence criterion of max force, default 0.01")  # fmt: skip
@click.option("-o", "--outfeather", default="all.feather", help="output, default all.feather")  # fmt: skip
def main(files, jobs, rundir, pressure, fmax, outfeather):
    rundir = Path(rundir)
    rundir.mkdir(exist_ok=True)
    finished = list(map(lambda p: p.stem, rundir.glob("*.xyz")))
    calc = get_calc_from_env()

    # backend = "multiprocessing"
    res = Parallel(jobs, verbose=2)(
        delayed(opt_one)(f, rundir, calc, pressure, fmax)
        for f in files
        if Path(f).stem not in finished
    )

    outfeather = Path(outfeather)
    if outfeather.exists():
        df = pd.read_feather(outfeather)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(res)]).reset_index(drop=True)
    print(df.drop("cif", axis=1, errors="ignore"))
    df.to_feather("all.feather")


if __name__ == "__main__":
    main()
