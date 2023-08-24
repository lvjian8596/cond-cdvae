# match and find unique structure in a given dir

import shutil
from collections import OrderedDict
from pathlib import Path

import click
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.vasp.inputs import Poscar
from tqdm import tqdm


@click.command
@click.argument("indir")
@click.argument("outdir")
def filter_uniq(indir, outdir):
    indir, outdir = Path(indir), Path(outdir)
    outdir.mkdir(exist_ok=True)
    flist = indir.glob("*.vasp")
    slist = OrderedDict({fname: Poscar.from_file(fname).structure for fname in flist})
    matcher = StructureMatcher()
    uniq = {}
    with tqdm(total=len(slist)) as pbar:
        while len(slist) > 0:
            pbar.update()
            k1, s1 = slist.popitem()
            for k2 in list(slist.keys()):
                if matcher.fit(s1, slist[k2]):
                    slist.pop(k2)
                    pbar.total -= 1
                    pbar.refresh()
            uniq[k1] = s1
            shutil.copy(k1, outdir)
    print(len(uniq))


if __name__ == '__main__':
    filter_uniq()
