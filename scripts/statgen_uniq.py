# find unique structures in a given dir

from collections import defaultdict
from multiprocessing import Pool, RLock, freeze_support
from pathlib import Path

import click
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from statgen import to_format_table
from tqdm import tqdm


def get_matchers(level):
    matcher_lo = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)  # loose
    matcher_md = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)  # midium
    matcher_st = StructureMatcher(ltol=0.1, stol=0.2, angle_tol=5)  # strict
    namelist = ["matcher_lo", "matcher_md", "matcher_st"]
    matcherlist = [matcher_lo, matcher_md, matcher_st]
    matchers = {n: m for n, m in zip(namelist[:level], matcherlist[:level])}
    return matchers


def get_uniq_df(idx, gendir, matchers):
    gen_list = sorted([int(f.stem) for f in gendir.glob("*.vasp")])

    genst_dict = {i: Structure.from_file(gendir / f"{i}.vasp") for i in gen_list}
    df = pd.DataFrame()
    for mat_name, matcher in matchers.items():
        pbar = tqdm(
            gen_list, ncols=160, desc=f"{gendir} {mat_name}", delay=1, position=idx
        )
        uniqid = defaultdict(list)
        for fi in pbar:
            formula = genst_dict[fi].composition.alphabetical_formula.replace(" ", "")
            df.loc[fi, "formula"] = formula
            for uid in uniqid[formula]:
                if matcher.fit(genst_dict[fi], genst_dict[uid]):
                    df.loc[fi, mat_name] = False
                    break
            else:
                df.loc[fi, mat_name] = True
                uniqid[formula].append(fi)
        pbar.close()

    df = df.sort_values("formula")
    table_str = to_format_table(df)
    with open(gendir.with_name("uniq.table"), "w") as f:
        f.write(table_str)

    return df


@click.command
@click.argument("gendirlist", nargs=-1)  # eval_gen*/gen
@click.option(
    "-l", "--level", default=1, type=int, help="number matcher, 1-2-3: loose-mid-strict"
)
@click.option("-j", "--njobs", default=-1, type=int, help="default: -1")
def filter_uniq(gendirlist, level, njobs):
    assert njobs > 1, "njobs must be larger than 1"
    matchers = get_matchers(level)
    gendirlist = [Path(d) for d in gendirlist if Path(d).is_dir()]

    freeze_support()
    pool = Pool(njobs, initializer=tqdm.set_lock, initargs=(RLock(),))
    for idx, gendir in enumerate(gendirlist):
        pool.apply_async(get_uniq_df, args=(idx, gendir, matchers))
    pool.close()
    pool.join()


if __name__ == '__main__':
    filter_uniq()
