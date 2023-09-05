# find unique structures in a given dir

import pickle
from pathlib import Path

import click
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from tqdm import tqdm
from statgen import to_format_table


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


def load_uniq_dict(picklefile):
    if Path(picklefile).exists():
        with open(picklefile, "rb") as f:
            uniq_dict = pickle.load(f)
    else:
        uniq_dict = {}
    return uniq_dict


def get_uniq_df(gendir, matchers, label):
    f_uniqtable = gendir.with_name(f"uniq.{label}.table")
    gen_list = sorted([int(f.stem) for f in gendir.glob("*.vasp")])

    if f_uniqtable.exists():
        df = pd.read_table(f_uniqtable, sep=r"\s+", index_col="index")
        if len(df) >= len(gen_list):
            return df

    genst_dict = {i: Structure.from_file(gendir / f"{i}.vasp") for i in gen_list}
    df = pd.DataFrame(True, gen_list, list(matchers.keys()))
    for mat_name, matcher in matchers.items():
        uniqid = []
        for fi in gen_list:
            for uid in uniqid:
                if matcher.fit(genst_dict[fi], genst_dict[uid]):
                    df.loc[fi, mat_name] = False
                    break
            else:
                uniqid.append(fi)

    table_str = to_format_table(df)
    with open(f_uniqtable, "w") as f:
        f.write(table_str)

    return df


@click.command
@click.argument("gendirlist", nargs=-1)  # eval_gen*/gen
@click.option("-j", "--njobs", default=-1, type=int, help="default: -1")
def filter_uniq(gendirlist, njobs):
    matchers = get_matchers()
    gendirlist = [Path(d) for d in gendirlist if Path(d).is_dir()]
    labellist = [d.parent.name[9:] for d in gendirlist]

    uniqdflist = Parallel(njobs, backend="multiprocessing")(
        delayed(get_uniq_df)(gendir, matchers, label)
        for gendir, label in tqdm(zip(gendirlist, labellist))
    )
    # uniq_dict = {mpname: df for mpname, df in zip(labellist, uniqdflist)}
    # pkl_dict = load_uniq_dict(picklefile)
    # pkl_dict.update(uniq_dict)
    # with open(picklefile, "wb") as f:
    #     pickle.dump(pkl_dict, f)


if __name__ == '__main__':
    filter_uniq()
