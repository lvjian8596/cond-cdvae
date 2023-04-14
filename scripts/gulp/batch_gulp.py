import time
import subprocess
from subprocess import PIPE
from pathlib import Path

import click
from joblib import Parallel, delayed
from tqdm import tqdm


def run(gin, rundir, retry, timeout):
    got = gin.with_suffix(".got")
    runflag = True
    while runflag and (retry > 0):
        # finished or not
        if got.exists():
            with open(got, 'r') as gotf:
                for line in gotf:
                    if "Finished" in line:
                        runflag = False
                        break
        # gulp < *.gin > *.got
        if runflag:
            with open(gin, 'r') as ginf, open(got, 'w') as gotf:
                try:
                    proc = subprocess.run(
                        ["gulp"],
                        stdin=ginf,
                        stdout=gotf,
                        stderr=PIPE,
                        cwd=rundir,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    retry = 1
                retry -= 1

    return runflag, gin


@click.command()
@click.option("-j", "--jobs", type=int, default=30, help="parallel threads, default 30")  # fmt: skip
@click.option("-r", "--rundir", default="rungulp", help="running dir including *.gin, default rungulp")  # fmt: skip
@click.option("--retry", default=5, type=int, help="max retry time of each structure, default 5")  # fmt: skip
@click.option("--timeout", default=10, type=float, help="max time of each opt, default 10")  # fmt: skip
def main(jobs, rundir, retry, timeout):
    ### find rungulp -name '*.got' |xargs grep -L 'Finished' |tee unopt.txt
    rundir = Path(rundir)
    res = Parallel(jobs)(
        delayed(run)(gin, rundir, retry, timeout)
        for gin in tqdm(list(rundir.glob("*.gin")), ncols=79)
    )
    for runflag, gin in res:
        if runflag:
            click.echo(f"Still failed: {gin}")


if __name__ == "__main__":
    main()
