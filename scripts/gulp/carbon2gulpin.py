import warnings
from pathlib import Path

import click
import numpy as np
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed
from tqdm import tqdm


def atoms2gulpcell(atoms):
    s = "Cell\n"
    s += "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(*atoms.cell.cellpar())
    return s


def atoms2gulpcoords(atoms):
    s = "fractional\n"
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
        s += "{} core {:9.7f} {:9.7f} {:9.7f}\n".format(symbol, pos[0], pos[1], pos[2])
    return s


def atoms2gulp(f, outdir, keywords, prange, nsample, maxcyc, lib, options):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = read(f)
    except Exception:
        click.echo(f"{f} read Error!")
    else:
        for samp, p in zip(range(nsample), np.random.uniform(*prange, nsample)):
            if nsample == 1:
                gin = outdir / f"{Path(f).stem}.gin"
            else:
                gin = outdir / f"{Path(f).stem}-{samp}.gin"
            with open(gin, "w") as ginf:
                ginf.write(" ".join(keywords) + "\n")
                ginf.write("\n")
                ginf.write(atoms2gulpcell(atoms))
                ginf.write(atoms2gulpcoords(atoms))
                ginf.write("\n")
                if lib.lower() != "none":
                    ginf.write(f"library {lib}\n")
                ginf.write(f"pressure {p}\n")
                ginf.write(f"maxcyc {maxcyc}\n")
                ginf.write(f"dump every {gin.stem}.gdp\n")
                ginf.write(f"output cif {gin.stem}.cif\n")
                ginf.write("\n".join(options))


@click.command(help="python atoms2gulpin.py -j 20 -o rungulp -k opti -k conp -p 0 50 -n 2 -op ... -op ...")  # fmt: skip
@click.argument("files", nargs=-1)  #, help="input files (*.cif, *.vasp, ...)")
@click.option("-j", "--jobs", default=10, type=int, help="max parllel jobs, default 10")
@click.option("-o", "--outdir", default="rungulp", help="out-directory, defualt rungulp")  # fmt: skip
@click.option("-k", "--keywords", multiple=True, default=("opti", "conjugate", "conp"), help="keywords, default opti conjugate conp")  # fmt: skip
@click.option("-p", "--prange", nargs=2, default=(0, 0), type=float, help="pressure range, default 0 0 (GPa)")  # fmt: skip
@click.option("-n", "--nsample", default=1, type=int, help="number of sample in prange, default 1")  # fmt: skip
@click.option("-l", "--lib", default="edip_marks.lib", help="library, won't write if given 'None', default edip_marks.lib") # fmt:# fmt: skip
@click.option("-c", "--maxcyc", default=850, type=int, help="default 850")
@click.option("-op", "--options", multiple=True, help="Other options in gulp")
def parallel_run(files, jobs, outdir, keywords, prange, nsample, lib, maxcyc, options):
    assert nsample >= 1
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    Parallel(jobs)(
        delayed(atoms2gulp)(f, outdir, keywords, prange, nsample, maxcyc, lib, options)
        for f in tqdm(files, ncols=79)
    )


if __name__ == "__main__":
    parallel_run()
