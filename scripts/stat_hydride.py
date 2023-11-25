import warnings
from itertools import chain
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


def rglobvasp(fdir: Path):
    fdir = Path(fdir)
    return list(
        chain(
            fdir.rglob("POSCAR"),
            fdir.rglob("POSCAR_*"),
            fdir.rglob("*.vasp"),
            fdir.rglob("poscar_*"),
            fdir.rglob("contcar_*"),
        )
    )


class HydrideAnalyzer:
    def __init__(self, structure: Structure, max_Hbond=1.0):
        self.CrystalNN = local_env.CrystalNN(
            x_diff_weight=-1,
            porous_adjustment=False,
        )
        self.structure = structure
        self.natoms = self.nsites = len(structure)
        self.formula = self.get_formula(structure, "hill")
        self.spgdict = self.get_spacegroup(structure)
        self.graph = self.get_graph(structure)
        self.distmat = self.get_distmat(structure)
        self.mindist = np.min(self.distmat)
        self.nH = self.get_nH(structure)
        self.nbondedH = self.get_nbondedH(structure, self.distmat, max_Hbond)

    @property
    def series(self):
        info = {}
        info.update(self.spgdict)
        keys_direct = ["formula", "nsites", "mindist", "nH", "nbondedH"]
        info.update({key: getattr(self, key) for key in keys_direct})
        return pd.Series(info)

    def get_spacegroup(self, structure):
        return {
            symkey: SpacegroupAnalyzer(structure, symprec).get_space_group_number()
            for symkey, symprec in zip(["5e-01", "1e-01", "1e-02"], [0.5, 0.1, 0.01])
        }

    def get_nH(self, structure: Structure):
        return sum([1 for n in (structure.atomic_numbers) if n == 1])

    def get_nbondedH(self, structure: Structure, distmat, max_Hbond=1.0):
        Hindices = [idx for idx, n in enumerate(structure.atomic_numbers) if n == 1]
        Hdistmat = np.take(distmat, Hindices, 0)
        return np.any(Hdistmat <= max_Hbond, axis=1).sum()

    def get_distmat(self, structure):
        distmat = structure.distance_matrix
        np.fill_diagonal(distmat, min(structure.lattice.abc))
        return distmat

    def get_formula(self, structure: Structure, mode="hill"):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        return atoms.get_chemical_formula(mode=mode)

    def get_graph(self, structure: Structure):
        """[[i, j, jix, jiy, jiz], ...]"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            g = StructureGraph.with_local_env_strategy(structure, self.CrystalNN).graph
        edges = [
            [i, j] + list(to_jimage) for i, j, to_jimage in g.edges(data='to_jimage')
        ]
        edges = np.array(edges, dtype=int)
        # reverse edge direction and flip to_jimage
        rev_edges = edges[:, [1, 0, 2, 3, 4]] * np.array([[1, 1, -1, -1, -1]])
        graph = np.concatenate([edges, rev_edges])
        return graph


def wrapped_analysis(fvasp, max_Hbond):
    structure = Poscar.from_file(fvasp).structure
    analyzer = HydrideAnalyzer(structure, max_Hbond)
    return analyzer.series


def analysis(njobs, vaspdirlist: list[Path], max_Hbond):
    for vaspdir in vaspdirlist:
        vaspflist = rglobvasp(vaspdir)
        # structures = [Poscar.from_file(fvasp).structure for fvasp in vaspflist]
        series_list = Parallel(njobs, "multiprocessing")(
            delayed(wrapped_analysis)(fvasp, max_Hbond)
            for fvasp in tqdm(vaspflist, desc=str(vaspdir)[-20:])
        )
        df = pd.DataFrame(series_list)
        print(df)


@click.command()
@click.argument('vaspdir', nargs=-1)
@click.option("-j", "--njobs", type=int, default=1)
@click.option("--max_hbond", type=float, default=1.0, help="max H bond (default 1.0)")
def main(vaspdir, njobs, max_hbond):
    analysis(njobs, vaspdir, max_Hbond)


if __name__ == "__main__":
    main()

