import subprocess
import time
import warnings
from pathlib import Path
from random import uniform
from subprocess import PIPE
from tempfile import TemporaryFile

import click
import pandas as pd
import tqdm
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed


def series2structure(series: pd.Series):
    # 1. ase Atoms
    with TemporaryFile('w+') as tmpfile:
        tmpfile.write(series.cif)
        tmpfile.seek(0)
        crystal = read(tmpfile, format='cif')
    return crystal


def run(gin, rundir):
    # gulp < *.gin > *.got
    got = gin.with_suffix(".got")
    with open(gin, 'r') as ginf, open(got, 'w') as gotf:
        proc = subprocess.run(
            ["gulp"],
            stdin=ginf,
            stdout=gotf,
            stderr=PIPE,
            cwd=rundir,
            text=True,
        )
    time.sleep(0.2)
    if len(proc.stderr) == 0:
        return True, gin.name
    else:
        return False, gin.name


@click.command()
@click.argument("infeather")
@click.option("-j", "--jobs", type=int, default=10)
@click.option("-r", "--rundir", default="rungulp", help="running dir including *.gin, default rungulp")  # fmt: skip
@click.option("-p", "--prange", nargs=2, default=(0, 0), type=float, help="pressure range, default 0 0 (GPa)")  # fmt: skip
def main(infeather, jobs, rundir, prange, maxsample):
    res = Parallel(jobs)(delayed()() for _ in _)


if __name__ == "__main__":
    main()
