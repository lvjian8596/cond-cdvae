import pickle
import subprocess
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure


def get_patched_mpds(mpds="/home/share/Data/MaterialsProject/mp-cif-230213.feather"):
    mpds = pd.read_feather(mpds)
    mpds.index = mpds.material_id
    return mpds


# [0, 20], [21, 40], [41, .]
def getnatomsgroup(mpds, material_id):
    natoms = mpds.loc[material_id, "nsites"]
    if natoms <= 20:
        return "0-20"
    elif natoms <= 40:
        return "21-40"
    else:
        return ">40"


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


# match *.vasp with ground-truth structure(gtst) with each matcher in matchers
# return
#   matcher_lo matcher_md matcher_st
# 0        T/F        T/F        T/F
# 1        ...        ...        ...
def match_genstructure(
    gendir: Path,  # */gen
    gtst: Structure,
    matchers: dict[str, StructureMatcher],
    label,
):
    f_target = gendir.with_name(f"{label}.vasp")
    f_matchtable = gendir.with_name(f"match.{label}.table")
    gen_list = sorted([int(f.stem) for f in gendir.glob("*.vasp")])

    if f_matchtable.exists():
        df = pd.read_table(f_matchtable, sep=r"\s+", index_col="index")
        if len(df) >= len(gen_list):
            return df

    genst_dict = {i: Structure.from_file(gendir / f"{i}.vasp") for i in gen_list}
    df = pd.DataFrame(
        {
            mat_name: pd.Series(
                {i: matcher.fit(gtst, genst) for i, genst in genst_dict.items()}
            )
            for mat_name, matcher in matchers.items()
        }
    )

    gtst.to(str(f_target), fmt="poscar")
    table_str = to_format_table(df)
    with open(f_matchtable, "w") as f:
        f.write(table_str)

    return df


def load_dictpkl(picklefile):
    if Path(picklefile).exists():
        with open(picklefile, "rb") as f:
            d = pickle.load(f)
    else:
        d = {}
    return d


def to_format_table(df: pd.DataFrame, index_label="index"):
    csv_str = df.to_csv(None, sep=" ", index_label=index_label)
    fmt_proc = subprocess.Popen(
        ['column', '-t'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    table_str, _ = fmt_proc.communicate(csv_str)
    return table_str
