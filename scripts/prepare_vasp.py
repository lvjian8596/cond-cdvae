import warnings
from pathlib import Path

import click
from joblib import Parallel, delayed
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import VaspInput
from pymatgen.io.vasp.sets import MPRelaxSet
from statgen import read_format_table
from tqdm import tqdm


def prepare_task(structure, relax_path, PSTRESS, NSW, sym, kspacing):
    user_incar_settings = {
        'NSW': NSW,
        'LREAL': False,
        'ISMEAR': 0,
        'EDIFF': 1e-6,
        'EDIFFG': -0.01,
        'PSTRESS': PSTRESS,
        'NCORE': 4,
        'ISYM': sym,
    }
    if kspacing is not None:
        user_incar_settings["KSPACING"] = kspacing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def wrapped_prepare_task(indir, uniq, uniqlevel, sf, nsw, pstress, sym, kspacing):
    runtype = ".scf" if nsw <= 1 else ".opt"
    if uniq is not None:
        runtype = f".uniq.{uniqlevel}" + runtype
    relax_path = indir.with_suffix(runtype).joinpath(sf.stem)
    relax_path.mkdir(exist_ok=True, parents=True)

    structure = Structure.from_file(sf)
    prepare_task(structure, relax_path, pstress, nsw, sym, kspacing)


@click.command
@click.argument("indir")
@click.option("-s", "--nsw", default=0, help="NSW, default 0")
@click.option("-p", "--pstress", default=0, help="PSTRESS (kbar), default 0")
@click.option("-ks", "--kspacing", type=float, help="KSPACING, default None")
@click.option("-u", "--uniq", default=None, help="unique file, default None")
@click.option(
    "-l",
    "--uniqlevel",
    default="lo",
    type=click.Choice(["lo", "md", "st"]),
    help="unique level of matcher, default lo",
)
@click.option("--sym", type=int, default=0, help="ISYM, default 0")
@click.option("-j", "--njobs", default=-1, type=int)
def prepare_batch(
    indir, nsw: int, pstress: float, njobs: int, uniq, uniqlevel, sym, kspacing
):
    click.echo(f"You are using {nsw=} {pstress=} {sym=} {kspacing=}")
    click.echo("Warning: W POTCAR is replaced by W_sv")
    indir = Path(indir)
    flist = list(indir.glob("*.vasp"))
    if uniq is not None:
        lv = f"matcher_{uniqlevel}"
        uniqdf = read_format_table(uniq)
        if lv not in uniqdf.columns:
            raise KeyError(f"key '{uniqlevel}' not in {uniq}")
        click.echo(f"using unique key '{lv}' in {uniq}")
        uniqlist = list(uniqdf[uniqdf[lv]].index)
        flist = [fi for fi in flist if int(fi.stem) in uniqlist]
    Parallel(njobs, backend="multiprocessing")(
        delayed(wrapped_prepare_task)(
            indir, uniq, uniqlevel, sf, nsw, pstress, sym, kspacing
        )
        for sf in tqdm(flist, ncols=120)
    )


if __name__ == '__main__':
    prepare_batch()
