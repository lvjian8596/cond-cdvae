# python ~/cond-cdvae/scripts/statgen_matchMPexpe.py eval_gen_mp-* -p match_dict.pkl
# >>>      matcher_lo  matcher_md  matcher_st
# >>> 1            40          14           6
# >>> 20          357         186         102
# >>> 40          487         258         167
#
# match_dict.pkl  # {mpname: matchdf}
# matchdf:
# |  matcher_lo  matcher_md  matcher_st|
# |i        T/F         T/F         T/F|

import warnings
from itertools import groupby
from pathlib import Path

import click
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core.structure import Structure
from statgen import (
    get_matchers,
    get_patched_mpds,
    getnatomsgroup,
    match_genstructure,
)
from tqdm import tqdm


def series2structure(series: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        crystal = Structure.from_str(series.cif, fmt='cif')
    return crystal


# get ground truth mp structure dict  {mpname: Structure}
def get_gtmp(mpname_list, mpds, njobs):
    gtmp_list = Parallel(njobs)(
        delayed(series2structure)(mpds.loc[mpname, :]) for mpname in tqdm(mpname_list)
    )
    gtmp = {mpname: st for mpname, st in zip(mpname_list, gtmp_list)}
    return gtmp


def getaccumdf(match_dict, splist):
    accum = []
    for sp in splist:
        ser = []
        for matchdf in match_dict.values():
            ser.append(matchdf[:sp].sum() > 0)
        accum.append(pd.Series(pd.DataFrame(ser).sum(), name=sp))
    return pd.DataFrame(accum)


def get_accumdf_dict(match_dict, mpds, step=20):
    splist = list(range(0, max([len(df) for df in match_dict.values()]) + 1, step))
    splist[0] = 1

    accumdf = getaccumdf(match_dict, splist)
    accumdf.loc["total"] = len(match_dict)
    accumdf_dict = {"accumdf_total": accumdf}

    for natomkey, mpidgroup in groupby(
        sorted(match_dict.keys(), key=lambda mpid: getnatomsgroup(mpds, mpid)),
        key=lambda mpid: getnatomsgroup(mpds, mpid),
    ):
        mpidgroup = list(mpidgroup)
        accumdf = getaccumdf({k: match_dict[k] for k in mpidgroup}, splist)
        accumdf.loc["total"] = len(mpidgroup)
        accumdf_dict[f"accumdf_{natomkey}"] = accumdf

    return accumdf_dict


@click.command
@click.argument("gendirlist", nargs=-1)  # eval_gen_*/gen
@click.option(
    "--mpds",
    default="/home/share/Data/MaterialsProject/mp-cif-230213.feather",
    help="Materials Project structures,"
    " default: /home/share/Data/MaterialsProject/mp-cif-230213.feather",
)
@click.option("-j", "--njobs", default=-1, help="default: -1")
def stat_mpexpe(gendirlist, mpds, njobs):
    click.echo("Loading MP dataset ...")
    mpds = get_patched_mpds(mpds)
    matchers = get_matchers()

    gendirlist = [Path(d) for d in gendirlist if Path(d).is_dir()]
    mpnamelist = [d.parent.name[9:] for d in gendirlist]

    gtmp = get_gtmp(mpnamelist, mpds, njobs)  # ground truth

    click.echo("Matching MP ...")
    match_list = Parallel(njobs, backend="multiprocessing")(
        delayed(match_genstructure)(gendir, gtmp[mpname], matchers, mpname)
        for mpname, gendir in tqdm(zip(mpnamelist, gendirlist))
    )
    return match_list


if __name__ == "__main__":
    stat_mpexpe()
