import re
import warnings
from pathlib import Path

import click
import pandas as pd
from ase.io import read
from ase.spacegroup import get_spacegroup
from joblib import Parallel, delayed
from tqdm import tqdm


def res2series(cif):
    # gulp dump file
    gdp = cif.with_suffix(".gdp")
    with open(gdp, "r") as gdpf:
        ENEPAT = re.compile(r"totalenergy\s+(-?\d+\.\d+)\s+eV")
        PRESPAT = re.compile(r"pressure\s+(-?\d+\.\d+)")
        ene, pres = None, None
        for line in gdpf.readlines():
            line = line.strip()
            if (ene is not None) and (pres is not None):
                break
            elif match := re.match(ENEPAT, line):
                ene = float(match.group(1))
            elif match := re.match(PRESPAT, line):
                pres = float(match.group(1))
        else:
            if pres is None:
                pres = 0.0
            else:
                print(gdp, "NOT MATCH")
            if ene is None:
                raise Exception(f"energy error: {cif}")
    # gulp output file
    got = cif.with_suffix(".got")
    with open(got, "r") as gotf:
        ENEPAT_GOT = re.compile(r".*\s+0 Energy:\s+(-?\d+\.\d+)")
        ene0 = None
        for line in gotf.readlines():
            line = line.strip()
            if ene0 is not None:
                break
            elif match := re.match(ENEPAT_GOT, line):
                ene0 = float(match.group(1))
        else:
            print(got, "NOT MATCH")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = read(cif)
        spgno = get_spacegroup(atoms).no
    with open(cif, "r") as ciff:
        cif = ciff.read()

    ser = pd.Series(
        {
            "material_id": gdp.stem,
            "nsites": len(atoms),
            "formula": atoms.get_chemical_formula("metal"),
            "spgno": spgno,
            "energy": ene,
            "energy_per_atom": ene / len(atoms),
            "energy0": ene0,
            "energy0_per_atom": ene0 / len(atoms),
            "pressure": pres,
            "volume": atoms.get_volume(),
            "cif": cif,
        }
    )
    return ser


@click.command()
@click.option("-r", "--rundir", default="rungulp", help="dir including *.gin *.got *.gdp *.cif, default rungulp")  # fmt: skip
@click.option("-j", "--jobs", default=10, type=int, help="max parallel jobs, default 10")  # fmt: skip
@click.option("-o", "--outfeather", default="all.feather", help="output in rundir, default all.feather")  # fmt: skip
@click.option("--debug", is_flag=True)
def parallel_run(rundir, jobs, outfeather, debug):
    rundir = Path(rundir)
    allcif = list(rundir.glob("*.cif"))
    if debug:
        allcif = allcif[:2]
    res = Parallel(jobs)(delayed(res2series)(cif) for cif in tqdm(allcif, ncols=79))
    df = pd.DataFrame(res).sort_values(by=["energy_per_atom"]).reset_index(drop=True)
    if debug:
        print(df.drop("cif", axis=1))
    else:
        df.to_feather(rundir / outfeather)


if __name__ == "__main__":
    parallel_run()
