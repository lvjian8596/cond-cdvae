# find the symmetry and standardlized cell of a given dir

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import click
import pandas as pd
import spglib
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed


def atoms2cif(atoms):
    with io.BytesIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='cif')
        cif = buffer.getvalue().decode()  # byte to string
    return cif


def atoms2vasp(atoms):
    with io.StringIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='vasp')
        vasp = buffer.getvalue()  # byte to string
    return vasp


def get_spg_one(name, atoms, symprec_list, angle_tolerance=10):
    spg_dict = {"name": str(name)}  # {name: str, prec: int}
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    for symprec in symprec_list:
        symds = spglib.get_symmetry_dataset(cell, symprec, angle_tolerance)
        if symds is not None:
            std_atoms = Atoms(
                symds["std_types"],
                cell=symds["std_lattice"],
                scaled_positions=symds["std_positions"],
            )
            spg_dict["{:.0e}".format(symprec)] = symds['number']
            spg_dict["{:.0e}".format(symprec) + "_std_natoms"] = len(symds['std_types'])
            spg_dict["{:.0e}".format(symprec) + "_std_cif"] = atoms2cif(std_atoms)
            spg_dict["{:.0e}".format(symprec) + "_std_vasp"] = atoms2vasp(std_atoms)
        else:
            print(name, symprec, "Cannot find symmetry", file=sys.stderr)
            spg_dict["{:.0e}".format(symprec)] = 0
            spg_dict["{:.0e}".format(symprec) + "_std_natoms"] = 0
            spg_dict["{:.0e}".format(symprec) + "_std_cif"] = atoms2cif(atoms)
            spg_dict["{:.0e}".format(symprec) + "_std_vasp"] = atoms2vasp(atoms)
    return pd.Series(spg_dict)


def get_spg(fdir, symprec_list=(0.5, 0.1, 0.01)):
    ser_list = Parallel(-1, backend="multiprocessing")(
        delayed(get_spg_one)(f, read(f), symprec_list) for f in Path(fdir).glob("*")
    )
    df = pd.DataFrame(ser_list)
    df = df.sort_values(by=list(map("{:.0e}".format, symprec_list)), ascending=False)
    return df


def to_format_csv(df: pd.DataFrame):
    df = df[[col for col in df if not col.endswith(("_std_cif", "_std_vasp"))]]
    csv_str = df.to_csv(None, sep=" ", index=False)
    fmt_proc = subprocess.Popen(
        ['column', '-t'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    output, _ = fmt_proc.communicate(csv_str)
    return output


def write_std_vasp(df: pd.DataFrame, indir):
    cols = [col for col in df if col.endswith("_std_vasp")]
    prec_list = [col[:-9] for col in cols]
    for prec in prec_list:
        prec_dir = Path(indir).with_name(f"std_{prec}")
        prec_dir.mkdir(exist_ok=True)
        for _, ser in df.iterrows():
            with open(prec_dir / Path(ser['name']).name, "w") as fvasp:
                fvasp.write(ser[prec + "_std_vasp"])


@click.command
@click.argument("indir")
@click.option(
    "-s",
    "--symprec",
    multiple=True,
    default=[0.5, 0.1, 0.01],
    help="symprec, can accept multiple time (not in one option)",
)
def cli_get_spg(indir, symprec):
    df = get_spg(indir, symprec)
    csv = to_format_csv(df)
    with open(Path(indir).with_name("spg.txt"), 'w') as f:
        f.write(csv)
    write_std_vasp(df, indir)


if __name__ == '__main__':
    cli_get_spg()
