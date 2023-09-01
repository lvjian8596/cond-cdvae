# python ~/cond-cdvae/scripts/stat_mpexpe.py eval_gen_mp-* -p match_dict.pkl
# >>>      matcher_lo  matcher_md  matcher_st
# >>> 1            40          14           6
# >>> 20          357         186         102
# >>> 40          487         258         167
#
# match_dict.pkl  # {mpname: matchdf}
# matchdf:
# |  matcher_lo  matcher_md  matcher_st|
# |i        T/F         T/F         T/F|

import pickle
import warnings
from itertools import groupby
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
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


def get_matchers():
    matcher_lo = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)  # loose
    matcher_md = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)  # midium
    matcher_st = StructureMatcher(ltol=0.1, stol=0.2, angle_tol=5)  # strict
    matchers = {
        "matcher_lo": matcher_lo,
        "matcher_md": matcher_md,
        "matcher_st": matcher_st,
    }
    return matchers


def match_genstructure(gendir, gtmp, matchers):
    mpname = gendir.name[9:]
    gen_list = sorted([int(f.stem) for f in gendir.joinpath("gen").glob("*.vasp")])
    genst_dict = {
        genfn.stem: Structure.from_file(genfn)
        for genfn in [gendir / f"gen/{i}.vasp" for i in gen_list]
    }
    gtst = gtmp[mpname]
    df = pd.DataFrame(
        {
            mat_name: pd.Series(
                {iname: matcher.fit(gtst, genst) for iname, genst in genst_dict.items()}
            )
            for mat_name, matcher in matchers.items()
        }
    )
    return df


# [0, 20], [21, 40], [41, .]
def getnatomsgroup(mpds, material_id):
    natoms = mpds.loc[material_id, "nsites"]
    if natoms <= 20:
        return "0-20"
    elif natoms <= 40:
        return "21-40"
    else:
        return ">40"


def getaccumdf(match_dict, splist):
    accum = []
    for sp in splist:
        ser = []
        for matchdf in match_dict.values():
            ser.append(matchdf[:sp].sum() > 0)
        accum.append(pd.Series(pd.DataFrame(ser).sum(), name=sp))
    return pd.DataFrame(accum)


def get_patched_mpds(mpds="/home/share/Data/MaterialsProject/mp-cif-230213.feather"):
    mpds = pd.read_feather(mpds)
    mpds.index = mpds.material_id
    return mpds


def load_match_dict(picklefile):
    if Path(picklefile).exists():
        with open(picklefile, "rb") as f:
            match_dict = pickle.load(f)
    else:
        match_dict = {}
    return match_dict


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
@click.argument("gendir", nargs=-1)
@click.option(
    "--mpds",
    default="/home/share/Data/MaterialsProject/mp-cif-230213.feather",
    help="Materials Project structures,"
    " default: /home/share/Data/MaterialsProject/mp-cif-230213.feather",
)
@click.option(
    "-p",
    "--picklefile",
    default="mpexpe_match_dict.pkl",
    help="out pickle file to update, default mpexpe_match_dict.pkl",
)
@click.option("-j", "--njobs", default=-1, help="default: -1")
def stat_mpexpe(gendir, mpds, picklefile, njobs):
    mpds = get_patched_mpds(mpds)
    match_dict = load_match_dict(picklefile)
    matchers = get_matchers()

    gendir = [Path(d) for d in gendir if Path(d).is_dir()]
    gendir = [d for d in gendir if d.name[9:] not in match_dict.keys()]
    genmpname = [d.name[9:] for d in gendir]

    gtmp = get_gtmp(genmpname, mpds, njobs)  # ground truth

    match_list = Parallel(njobs)(
        delayed(match_genstructure)(d, gtmp, matchers) for d in tqdm(gendir)
    )
    match_dict = {
        **match_dict,
        **{mpname: matchdf for mpname, matchdf in zip(genmpname, match_list)},
    }
    with open(picklefile, "wb") as f:
        pickle.dump(match_dict, f)


if __name__ == "__main__":
    stat_mpexpe()
